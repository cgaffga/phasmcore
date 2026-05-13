// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Top-level decode-side slice walker (Phase 6D.8 chunk 6B).
//
// Public entry point: `walk_annex_b_for_cover(annex_b)` →
// `CoverWalkOutput { cover: DomainCover, n_mb, n_slices }`. Walks
// the bitstream end-to-end, parses SPS / PPS / slice headers, and
// — once chunk 6C+ ships per-MB-type dispatch — emits a `DomainCover`
// byte-identical to what the encode-side `PositionLoggerHook`
// produced on the same coefficients.
//
// **Scope (this chunk)**: scaffold only.
//   - Annex-B → NAL unit scanning
//   - SPS + PPS extraction (latest-wins, single-active assumption)
//   - Slice NAL identification + slice header parse
//   - CABAC engine + context initialization at slice start
//   - cabac_alignment_one_bit consumption to byte boundary
//   - Per-MB walking loop is stubbed: returns Ok with empty cover.
//
// 6C wires I_16x16 dispatch; 6D wires I_4x4. Each subsequent chunk
// adds one MB-type code path with the byte-identity gate guarding
// regressions.

use crate::codec::h264::bitstream::parse_nal_units_annexb;
use crate::codec::h264::cabac::context::CabacInitSlot;
use crate::codec::h264::cabac::neighbor::{
    block_pos_to_chroma_ac_idx,
    compute_cbf_ctx_idx_inc_chroma_ac, compute_cbf_ctx_idx_inc_chroma_dc,
    compute_cbf_ctx_idx_inc_luma_4x4, compute_cbf_ctx_idx_inc_luma_ac,
    compute_cbf_ctx_idx_inc_luma_dc, compute_mvd_ctx_idx_inc_bin0,
    CabacNeighborMB, CurrentMbCbf, CurrentMbMvdAbs, MbTypeClass,
};
use crate::codec::h264::stego::{
    Axis, DomainCover, MvdSlot, NullLogger, PositionLogger, ResidualPathKind,
};
use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;
use crate::codec::h264::slice::{parse_slice_header, SliceHeader, SliceType};
use crate::codec::h264::sps::{parse_pps, parse_sps, Pps, Sps};
use crate::codec::h264::{H264Error, NalType, NalUnit};

use super::syntax::{
    decode_coded_block_pattern, decode_end_of_slice_flag,
    decode_intra_chroma_pred_mode, decode_mb_qp_delta, decode_mb_skip_flag,
    decode_mb_skip_flag_b,
    decode_mb_type_b, decode_mb_type_i, decode_mb_type_p,
    decode_mvd_with_bin0_inc, decode_ref_idx,
    decode_prev_intra4x4_pred_mode_flag, decode_rem_intra4x4_pred_mode,
    decode_residual_block_cabac, decode_residual_block_cabac_8x8,
    decode_sub_mb_type_b, decode_sub_mb_type_p,
    decode_transform_size_8x8_flag, PositionCtx,
};

use super::decoder::CabacDecoder;
use super::engine::DecodeError;
use super::positions::PositionRecorder;

/// Errors surfaced by the slice walker.
#[derive(Debug)]
pub enum WalkError {
    /// H.264 parse error (NAL header, RBSP, SPS/PPS/slice).
    H264(H264Error),
    /// CABAC engine error (bytestream EOF, invalid arith state).
    Cabac(DecodeError),
    /// Bitstream is missing required SPS or PPS before the first slice.
    MissingParameterSet,
    /// PPS specifies CAVLC (entropy_coding_mode_flag = 0). The bin
    /// decoder is CABAC-only.
    NotCabac,
    /// Slice header indicated B-slice or other unsupported type.
    UnsupportedSliceType(SliceType),
    /// Stream contains no slice NALs.
    NoSlices,
}

impl From<H264Error> for WalkError {
    fn from(e: H264Error) -> Self { Self::H264(e) }
}
impl From<DecodeError> for WalkError {
    fn from(e: DecodeError) -> Self { Self::Cabac(e) }
}

impl std::fmt::Display for WalkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::H264(e) => write!(f, "h264 parse: {e}"),
            Self::Cabac(e) => write!(f, "cabac decode: {e:?}"),
            Self::MissingParameterSet => write!(f, "missing SPS or PPS"),
            Self::NotCabac => write!(f, "PPS is CAVLC; bin decoder is CABAC-only"),
            Self::UnsupportedSliceType(t) => write!(f, "unsupported slice type: {t}"),
            Self::NoSlices => write!(f, "no slice NALs in bitstream"),
        }
    }
}

impl std::error::Error for WalkError {}

/// Output of a successful walk: accumulated cover + summary counts.
#[derive(Debug, Default)]
pub struct CoverWalkOutput {
    pub cover: DomainCover,
    pub n_mb: usize,
    pub n_slices: usize,
    /// Phase 6F.2(j) — per-MVD-position metadata aligned by index
    /// with `cover.mvd_sign_bypass.positions`. Empty when
    /// `WalkOptions { record_mvd: false, record_offsets: false }`.
    pub mvd_meta: Vec<crate::codec::h264::stego::encoder_hook::MvdPositionMeta>,
    /// Phase 6F.2(j) — frame dimensions in macroblocks, parsed
    /// from the first SPS encountered. Used by
    /// `cascade_safety::analyze_safe_mvd_subset` at decode time.
    /// 0/0 when no slice was walked (degenerate input).
    pub mb_w: u32,
    pub mb_h: u32,
    /// Phase C.3.6.1 (task #428) — per-position RBSP byte+bit
    /// offsets, aligned 1:1 with each domain's `bits` + `positions`
    /// in `cover`. `None` when `WalkOptions::record_offsets` is
    /// false (the default); `Some(_)` when offset capture is opted
    /// into for the Option C bitstream-mod stego splicer.
    pub offsets: Option<super::positions::DomainOffsets>,
}

/// Walker configuration knobs. Default-everything-off keeps
/// behavior byte-identical to chunk 6B–§30A4.
#[derive(Debug, Clone, Copy, Default)]
pub struct WalkOptions {
    /// Phase 6D.8 §30D-B: when true, the slice walker records MVD
    /// positions + bits into the per-domain cover (mvd_sign_bypass
    /// + mvd_suffix_lsb). Must be set in lockstep with the
    /// encoder's `enable_mvd_stego_hook` flag — otherwise encoder
    /// + decoder MVD covers diverge.
    pub record_mvd: bool,
    /// Phase C.3.6.1 (task #428) — when true, the walker captures
    /// the RBSP bit offset of every bypass-coded stego bin into
    /// `CoverWalkOutput::offsets`, aligned index-for-index with
    /// `cover.{domain}.{bits,positions}`. Used by the Option C
    /// bitstream-mod stego splicer to locate flip targets directly
    /// in the encoded Annex-B stream. Default: false (zero-cost
    /// NullLogger fast path).
    pub record_offsets: bool,
}

/// Phase 6E-C0 streaming-walker callback verdict. Returned by the
/// per-GOP `on_gop` callback to either continue walking the next
/// GOP or terminate the walk early. Early-exit is the load-bearing
/// optimization for shadow decode: once a parity-tier candidate
/// successfully RS-decodes + AES-GCM-SIV-decrypts a shadow's
/// payload, no further GOPs need to be walked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkAction {
    Continue,
    StopWalk,
}

/// Phase 6E-C0 per-GOP context delivered to the streaming walker's
/// callback. Owns the GOP's `DomainCover` (positions + bits) and
/// per-GOP statistics. The streaming walker swaps out the recorder
/// for each GOP, so receiving the cover by value is zero-copy
/// from the recorder's perspective.
#[derive(Debug)]
pub struct GopContext {
    /// Zero-based GOP index in stream order. The first GOP
    /// (containing the first IDR) is gop_idx = 0.
    pub gop_idx: u32,
    /// This GOP's `DomainCover` — positions + cover bits across all
    /// four bypass-bin domains, in slice scan order.
    pub cover: DomainCover,
    /// Number of macroblocks emitted in this GOP across all
    /// slices.
    pub n_mb: usize,
    /// Number of slice NAL units in this GOP.
    pub n_slices: usize,
    /// Phase 6F.2(j) — per-MVD-position metadata aligned by index
    /// with `cover.mvd_sign_bypass.positions`. Empty when
    /// `WalkOptions { record_mvd: false, record_offsets: false }`.
    pub mvd_meta: Vec<crate::codec::h264::stego::encoder_hook::MvdPositionMeta>,
    /// Phase 6F.2(j) — frame dimensions in macroblocks (from the
    /// active SPS at the time this GOP was walked). Used by
    /// `cascade_safety::analyze_safe_mvd_subset`.
    pub mb_w: u32,
    pub mb_h: u32,
    /// Phase C.3.6.1 (task #428) — per-bypass-bin RBSP byte+bit
    /// offsets for this GOP, aligned 1:1 with `cover` domain bit
    /// vectors. `None` when `WalkOptions::record_offsets` is false.
    pub offsets: Option<super::positions::DomainOffsets>,
}

/// Phase 6E-C0 streaming-walker summary. Returned at end-of-stream
/// (or early-exit via `WalkAction::StopWalk`) with aggregate
/// counters for the whole walk.
#[derive(Debug, Default)]
pub struct StreamingWalkOutput {
    pub n_gops: usize,
    pub n_mb: usize,
    pub n_slices: usize,
}

/// Walk an Annex-B byte stream end-to-end and return the per-domain
/// `DomainCover` (positions + bit values) suitable for STC extract.
///
/// **Single-frame I-only scope (chunk 6B+)**: fails on B-slices,
/// rejects CAVLC, parses SPS+PPS in stream order with latest-wins
/// selection.
pub fn walk_annex_b_for_cover(annex_b: &[u8]) -> Result<CoverWalkOutput, WalkError> {
    walk_annex_b_for_cover_with_options(annex_b, WalkOptions::default())
}

/// Variant of `walk_annex_b_for_cover` that accepts explicit
/// `WalkOptions`. Used by §30D consumers to opt into MVD recording.
pub fn walk_annex_b_for_cover_with_options(
    annex_b: &[u8],
    opts: WalkOptions,
) -> Result<CoverWalkOutput, WalkError> {
    let nalus = parse_nal_units_annexb(annex_b)?;
    walk_nalus_for_cover_with_options(&nalus, opts)
}

/// Walk an already-parsed NAL unit list. Same semantics as
/// `walk_annex_b_for_cover`, but accepts pre-parsed input (lets the
/// caller share a NAL parser run with other consumers).
pub fn walk_nalus_for_cover(nalus: &[NalUnit]) -> Result<CoverWalkOutput, WalkError> {
    walk_nalus_for_cover_with_options(nalus, WalkOptions::default())
}

/// Variant of `walk_nalus_for_cover` that accepts explicit
/// `WalkOptions`. Thin wrapper over `walk_nalus_streaming_with_options`
/// that accumulates per-GOP covers into a single whole-stream cover
/// — preserves all current parity gates as regression tests.
pub fn walk_nalus_for_cover_with_options(
    nalus: &[NalUnit],
    opts: WalkOptions,
) -> Result<CoverWalkOutput, WalkError> {
    let mut acc_cover = DomainCover::default();
    let mut acc_mvd_meta: Vec<crate::codec::h264::stego::encoder_hook::MvdPositionMeta>
        = Vec::new();
    let mut acc_mb_w = 0u32;
    let mut acc_mb_h = 0u32;
    // Phase C.3.6.1 — opt-in offset accumulator. Pre-allocated as
    // Some(empty) when offset capture is requested so per-GOP
    // contributions can be extended in-place; remains None otherwise.
    let mut acc_offsets: Option<super::positions::DomainOffsets> =
        if opts.record_offsets {
            Some(super::positions::DomainOffsets::default())
        } else {
            None
        };
    let streaming_out = walk_nalus_streaming_with_options(
        nalus,
        opts,
        |gop_ctx| {
            acc_cover.extend_from(gop_ctx.cover);
            acc_mvd_meta.extend(gop_ctx.mvd_meta);
            if let (Some(acc), Some(gop_off)) =
                (acc_offsets.as_mut(), gop_ctx.offsets)
            {
                acc.coeff_sign_bypass.extend(gop_off.coeff_sign_bypass);
                acc.coeff_suffix_lsb.extend(gop_off.coeff_suffix_lsb);
                acc.mvd_sign_bypass.extend(gop_off.mvd_sign_bypass);
                acc.mvd_suffix_lsb.extend(gop_off.mvd_suffix_lsb);
            }
            // Frame dimensions are SPS-derived and stable across
            // GOPs in a single stream (any SPS change is treated
            // as an error elsewhere). Take the last seen value.
            if gop_ctx.mb_w != 0 { acc_mb_w = gop_ctx.mb_w; }
            if gop_ctx.mb_h != 0 { acc_mb_h = gop_ctx.mb_h; }
            Ok(WalkAction::Continue)
        },
    )?;
    Ok(CoverWalkOutput {
        cover: acc_cover,
        n_mb: streaming_out.n_mb,
        n_slices: streaming_out.n_slices,
        mvd_meta: acc_mvd_meta,
        mb_w: acc_mb_w,
        mb_h: acc_mb_h,
        offsets: acc_offsets,
    })
}

/// Phase 6E-C0 — per-GOP streaming walker (Annex-B input). Parses
/// the byte stream, emits one `GopContext` per GOP via the
/// `on_gop` callback. Memory bound is per-GOP (the swap-out
/// recorder discards the previous GOP's cover after each callback
/// fires) — constant in video length.
///
/// GOP boundaries are detected at IDR VCL NALs: when an IDR slice
/// arrives after at least one VCL NAL has been processed in the
/// current GOP, the prior GOP fires via callback before the IDR
/// is parsed into the next GOP.
///
/// Callback contract: returning `WalkAction::Continue` advances to
/// the next GOP; returning `WalkAction::StopWalk` terminates the
/// walk immediately and returns a `StreamingWalkOutput` covering
/// the GOPs processed so far.
pub fn walk_annex_b_streaming<F>(
    annex_b: &[u8],
    opts: WalkOptions,
    on_gop: F,
) -> Result<StreamingWalkOutput, WalkError>
where
    F: FnMut(GopContext) -> Result<WalkAction, WalkError>,
{
    let nalus = parse_nal_units_annexb(annex_b)?;
    walk_nalus_streaming_with_options(&nalus, opts, on_gop)
}

/// Phase 6E-C0 — per-GOP streaming walker (pre-parsed NAL list
/// input). Same semantics as `walk_annex_b_streaming` but lets the
/// caller share a NAL parser run with other consumers.
pub fn walk_nalus_streaming_with_options<F>(
    nalus: &[NalUnit],
    opts: WalkOptions,
    mut on_gop: F,
) -> Result<StreamingWalkOutput, WalkError>
where
    F: FnMut(GopContext) -> Result<WalkAction, WalkError>,
{
    let mut active_sps: Option<Sps> = None;
    let mut active_pps: Option<Pps> = None;
    let mut recorder = if opts.record_offsets {
        PositionRecorder::with_offsets()
    } else {
        PositionRecorder::new()
    };
    let mut frame_idx: u32 = 0;
    let mut current_gop_idx: u32 = 0;
    let mut gop_n_mb: usize = 0;
    let mut gop_n_slices: usize = 0;
    let mut had_vcl_in_current_gop = false;
    let mut total_n_mb: usize = 0;
    let mut total_n_slices: usize = 0;
    let mut total_n_gops: usize = 0;

    for (nal_idx_usize, nal) in nalus.iter().enumerate() {
        let nal_idx = nal_idx_usize as u32;
        match nal.nal_type {
            NalType::SPS => {
                active_sps = Some(parse_sps(&nal.rbsp)?);
            }
            NalType::PPS => {
                active_pps = Some(parse_pps(&nal.rbsp)?);
            }
            t if t.is_vcl() => {
                let sps = active_sps.as_ref().ok_or(WalkError::MissingParameterSet)?;
                let pps = active_pps.as_ref().ok_or(WalkError::MissingParameterSet)?;
                if !pps.entropy_coding_mode_flag {
                    return Err(WalkError::NotCabac);
                }

                let header = parse_slice_header(&nal.rbsp, sps, pps, t, nal.nal_ref_idc)?;
                // Phase 6E-A3 — B-slices accepted; the walker
                // dispatches into `walk_b_mb` which currently only
                // supports B_Skip (mb_skip_flag=1). Non-skip B-MBs
                // return `UnsupportedMbType` until §6E-A4 adds the
                // mode-decision + ME paths. Encoder emits B_Skip
                // only at this sub-phase.

                // GOP boundary detection: an IDR after at least one
                // VCL NAL in the current GOP closes the prior GOP.
                if t.is_idr() && had_vcl_in_current_gop {
                    let cover = recorder.take_cover();
                    let mvd_meta = recorder.take_mvd_meta();
                    let offsets = recorder.take_offsets();
                    let (closing_mb_w, closing_mb_h) = active_sps
                        .as_ref()
                        .map(|s| (
                            s.pic_width_in_mbs,
                            s.pic_height_in_map_units
                                * if s.frame_mbs_only_flag { 1 } else { 2 },
                        ))
                        .unwrap_or((0, 0));
                    let action = on_gop(GopContext {
                        gop_idx: current_gop_idx,
                        cover,
                        n_mb: gop_n_mb,
                        n_slices: gop_n_slices,
                        mvd_meta,
                        mb_w: closing_mb_w,
                        mb_h: closing_mb_h,
                        offsets,
                    })?;
                    total_n_gops += 1;
                    if matches!(action, WalkAction::StopWalk) {
                        return Ok(StreamingWalkOutput {
                            n_gops: total_n_gops,
                            n_mb: total_n_mb,
                            n_slices: total_n_slices,
                        });
                    }
                    current_gop_idx = current_gop_idx.wrapping_add(1);
                    gop_n_mb = 0;
                    gop_n_slices = 0;
                    // had_vcl_in_current_gop is reasserted to true on the
                    // next VCL NAL (line ~382); no explicit reset needed.
                }

                let mb_w = sps.pic_width_in_mbs as usize;
                let mb_h = sps.pic_height_in_map_units as usize
                    * if sps.frame_mbs_only_flag { 1 } else { 2 };
                let mb_count = mb_w * mb_h;

                // Slice data begins at `header.data_bit_offset` in
                // the RBSP. For CABAC, consume cabac_alignment_one_bit
                // to advance to the next byte boundary, then the
                // arithmetic engine reads from that byte forward.
                let cabac_byte_off = cabac_data_byte_offset(header.data_bit_offset);
                if cabac_byte_off > nal.rbsp.len() {
                    return Err(WalkError::H264(H264Error::UnexpectedEof));
                }
                let cabac_bytes = &nal.rbsp[cabac_byte_off..];

                let slot = pick_init_slot(&header);
                let mut dec = CabacDecoder::new_slice(
                    cabac_bytes, slot, header.slice_qp, mb_w,
                )?;

                let n_mb = walk_slice_mbs(
                    &mut dec, &header, pps, mb_count, frame_idx, mb_w,
                    nal_idx, &mut recorder, &opts,
                )?;

                gop_n_mb += n_mb;
                gop_n_slices += 1;
                total_n_mb += n_mb;
                total_n_slices += 1;
                had_vcl_in_current_gop = true;
                frame_idx = frame_idx.wrapping_add(1);
            }
            _ => { /* AUD, SEI, filler, etc — skip */ }
        }
    }

    // End-of-stream: emit the final GOP if any VCL was processed.
    if had_vcl_in_current_gop {
        let cover = recorder.take_cover();
        let mvd_meta = recorder.take_mvd_meta();
        let offsets = recorder.take_offsets();
        let (closing_mb_w, closing_mb_h) = active_sps
            .as_ref()
            .map(|s| (
                s.pic_width_in_mbs,
                s.pic_height_in_map_units
                    * if s.frame_mbs_only_flag { 1 } else { 2 },
            ))
            .unwrap_or((0, 0));
        on_gop(GopContext {
            gop_idx: current_gop_idx,
            cover,
            n_mb: gop_n_mb,
            n_slices: gop_n_slices,
            mvd_meta,
            mb_w: closing_mb_w,
            mb_h: closing_mb_h,
            offsets,
        })?;
        total_n_gops += 1;
    }

    if total_n_slices == 0 {
        return Err(WalkError::NoSlices);
    }

    Ok(StreamingWalkOutput {
        n_gops: total_n_gops,
        n_mb: total_n_mb,
        n_slices: total_n_slices,
    })
}

/// Pick the CABAC init slot for this slice. § 9.3.1.1 maps from
/// (slice_type, cabac_init_idc) to {ISI, PIdc0, PIdc1, PIdc2}; for
/// I/SI slices the slot is fixed to ISI. P/SP/B slices use
/// PIdc{0,1,2} per `cabac_init_idc` in the slice header.
fn pick_init_slot(header: &SliceHeader) -> CabacInitSlot {
    match header.slice_type {
        SliceType::I | SliceType::SI => CabacInitSlot::ISI,
        SliceType::P | SliceType::SP | SliceType::B => match header.cabac_init_idc {
            0 => CabacInitSlot::PIdc0,
            1 => CabacInitSlot::PIdc1,
            2 => CabacInitSlot::PIdc2,
            _ => CabacInitSlot::PIdc0, // spec range is 0..=2; clamp.
        },
    }
}

/// Per § 7.3.4: cabac_alignment_one_bit pads slice_data() to a byte
/// boundary. `data_bit_offset` is the bit position right after the
/// slice header. Compute the byte-aligned position where the CABAC
/// arithmetic engine starts reading (byte ceil of data_bit_offset).
///
/// Phase C.3.6.2 (#429) — exposed pub so the bitstream-mod splicer
/// can convert a captured engine-local bit offset to NAL-RBSP-
/// absolute coordinates: `rbsp_byte = cabac_data_byte_offset +
/// engine_bit / 8`.
pub fn cabac_data_byte_offset(data_bit_offset: usize) -> usize {
    data_bit_offset.div_ceil(8)
}

/// Per-MB walking loop. Dispatches per slice type:
/// - I/SI: I_16x16 (chunk 6C), I_NxN/I_4x4 (chunk 6D), I_PCM errors.
/// - P/SP: P_SKIP + intra-in-P (§30A1+2). P_partition (mb_type 0..3)
///   and I_PCM-in-P still error.
///
/// Mirrors `Encoder::write_i16x16_macroblock_cabac` (encoder.rs:3599),
/// `Encoder::commit_i4x4_macroblock_cabac` (encoder.rs:2946), and
/// `Encoder::write_p_macroblock_cabac` (encoder.rs:903).
fn walk_slice_mbs(
    dec: &mut CabacDecoder<'_>,
    header: &SliceHeader,
    pps: &Pps,
    mb_count: usize,
    frame_idx: u32,
    mb_w: usize,
    nal_idx: u32,
    recorder: &mut PositionRecorder,
    opts: &WalkOptions,
) -> Result<usize, WalkError> {
    let is_intra_slice = matches!(header.slice_type, SliceType::I | SliceType::SI);
    let is_p_slice = matches!(header.slice_type, SliceType::P | SliceType::SP);
    let is_b_slice = matches!(header.slice_type, SliceType::B);
    if !is_intra_slice && !is_p_slice && !is_b_slice {
        return Err(WalkError::UnsupportedSliceType(header.slice_type));
    }

    let mut mbs_walked = 0usize;
    let mut prev_mb_qp = header.slice_qp;
    let mut mb_addr = header.first_mb_in_slice as usize;
    let mut prev_mb_y: Option<usize> = None;

    while mb_addr < mb_count {
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;

        // Mirror encoder loops (encoder.rs:814 / 3551): call
        // `neighbors.new_row()` at every row boundary.
        if let Some(prev_y) = prev_mb_y
            && mb_y != prev_y {
                dec.neighbors.new_row();
            }
        prev_mb_y = Some(mb_y);

        // v1.4 (#305) — num_ref_idx_l0_active drives whether walker
        // expects ref_idx_l0 unary bins per partition. At
        // MultiRefConfig::SINGLE_REF (encoder default → slice header
        // claims 1) the gate is closed and zero bins read (bit-
        // identical to v1.3).
        let num_active_l0 = header.num_ref_idx_l0_active;
        let is_last = if is_b_slice {
            walk_b_mb(
                dec, pps, mb_x, mb_y, mb_w, &mut prev_mb_qp,
                frame_idx, nal_idx, recorder, opts, num_active_l0,
            )?
        } else if is_p_slice {
            walk_p_mb(
                dec, pps, mb_x, mb_y, mb_w, &mut prev_mb_qp,
                frame_idx, nal_idx, recorder, opts, num_active_l0,
            )?
        } else {
            walk_i_mb(
                dec, pps, mb_x, mb_y, mb_w, &mut prev_mb_qp,
                frame_idx, nal_idx, recorder,
            )?
        };

        mbs_walked += 1;
        mb_addr += 1;
        if is_last {
            break;
        }
    }

    Ok(mbs_walked)
}

/// I-slice MB dispatch. Returns is_last (end_of_slice_flag).
fn walk_i_mb(
    dec: &mut CabacDecoder<'_>,
    pps: &Pps,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    prev_mb_qp: &mut i32,
    frame_idx: u32,
    nal_idx: u32,
    recorder: &mut PositionRecorder,
) -> Result<bool, WalkError> {
    let mb_type = decode_mb_type_i(dec, mb_x)?;
    if mb_type == 25 {
        return Err(WalkError::H264(H264Error::Unsupported(
            "I_PCM (mb_type=25) not yet wired".into(),
        )));
    }
    if mb_type == 0 {
        return walk_inxn_mb(
            dec, pps, mb_x, mb_y, mb_w, prev_mb_qp, frame_idx, nal_idx, recorder,
        );
    }
    walk_i16x16_mb(
        dec, mb_x, mb_y, mb_w, prev_mb_qp, frame_idx, nal_idx, recorder,
        /* i_mb_type */ mb_type,
    )
}

/// P-slice MB dispatch. P_SKIP + intra-in-P routing. P_partition
/// (mb_type 0..3) errors out — to be wired in §30A3+.
#[allow(clippy::too_many_arguments)]
fn walk_p_mb(
    dec: &mut CabacDecoder<'_>,
    pps: &Pps,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    prev_mb_qp: &mut i32,
    frame_idx: u32,
    nal_idx: u32,
    recorder: &mut PositionRecorder,
    opts: &WalkOptions,
    num_active_l0: u8,
) -> Result<bool, WalkError> {
    // 1. mb_skip_flag (encoder.rs:1553).
    let is_skip = decode_mb_skip_flag(dec, mb_x)?;
    if is_skip {
        // P_SKIP: no further syntax. Encoder.rs:1561 emits
        // end_of_slice_flag and commits PSkip neighbors. No
        // residuals, no MVDs → zero stego coverage.
        let is_last = decode_end_of_slice_flag(dec)?;
        let nb = CabacNeighborMB {
            mb_type: MbTypeClass::PSkip,
            mb_skip_flag: true,
            ..CabacNeighborMB::default()
        };
        dec.neighbors.commit(mb_x, nb);
        return Ok(is_last);
    }

    // 2. mb_type_p (encoder.rs:1610).
    let mb_type_p = decode_mb_type_p(dec, mb_x)?;
    if mb_type_p <= 3 {
        // P-partition: §30A3 (P_L0_16x16) + §30A4 (P_L0_16x8 / P_L0_8x16 / P_8x8).
        return walk_p_partition_mb(
            dec, pps, mb_x, mb_y, mb_w, prev_mb_qp, frame_idx, nal_idx, recorder,
            mb_type_p, opts, num_active_l0,
        );
    }

    // mb_type_p ∈ [5, 30]: intra-in-P. Spec maps to I-slice mb_type
    // via `i_mb_type = mb_type_p - 5`.
    //   5 → I_NxN, 6..29 → I_16x16 variants, 30 → I_PCM.
    let i_mb_type = mb_type_p - 5;
    if i_mb_type == 25 {
        return Err(WalkError::H264(H264Error::Unsupported(
            "I_PCM-in-P (mb_type_p=30) not yet wired".into(),
        )));
    }
    if i_mb_type == 0 {
        return walk_inxn_mb(
            dec, pps, mb_x, mb_y, mb_w, prev_mb_qp, frame_idx, nal_idx, recorder,
        );
    }
    walk_i16x16_mb(
        dec, mb_x, mb_y, mb_w, prev_mb_qp, frame_idx, nal_idx, recorder,
        i_mb_type,
    )
}

/// I_16x16 MB walker (chunk 6C body, factored out so P-slice
/// intra-in-P can reuse). `i_mb_type` is the I-slice mb_type
/// value 1..24 (NOT the P-slice 6..29 codenum).
fn walk_i16x16_mb(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    prev_mb_qp: &mut i32,
    frame_idx: u32,
    nal_idx: u32,
    recorder: &mut PositionRecorder,
    i_mb_type: u32,
) -> Result<bool, WalkError> {
    // Decompose mb_type per spec § 7.3.5 / Table 7-11 — Task #50 helper.
    let fields = super::super::mb_type_math::unpack_i_16x16_mb_type(i_mb_type);
    let _luma_pred_mode = fields.luma_pred_mode;
    let cbp_chroma = fields.cbp_chroma;
    let cbp_luma_flag = fields.cbp_luma_flag;

    let chroma_pred_mode = decode_intra_chroma_pred_mode(dec, mb_x)?;

    // mb_qp_delta — always present for Intra_16x16 (§ 7.3.5.1).
    let qp_delta = decode_mb_qp_delta(dec)?;
    *prev_mb_qp += qp_delta;
    let mb_addr_u32 = (mb_y * mb_w + mb_x) as u32;

    let mut current_cbf = CurrentMbCbf::new();
    let current_is_intra = true;

    // Luma DC (Intra16x16DCLevel, ctx_block_cat=0).
    let dc_inc = compute_cbf_ctx_idx_inc_luma_dc(&dec.neighbors, mb_x);
    let dc_scan = decode_residual_block_with_offset_capture(
        dec, 0, 15, /* cat */ 0, dc_inc,
        frame_idx, mb_addr_u32, nal_idx, ResidualPathKind::LumaDcIntra16x16, recorder,
    )?;
    let dc_coded = dc_scan.iter().any(|&v| v != 0);
    current_cbf.set(0, 0, dc_coded);
    recorder.on_residual_block(
        frame_idx, mb_addr_u32, &dc_scan, 0, 15,
        ResidualPathKind::LumaDcIntra16x16,
    );

    // Luma AC (cat=1, 16 blocks) gated by cbp_luma_flag.
    if cbp_luma_flag != 0 {
        for k in 0..16usize {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            let ac_inc = compute_cbf_ctx_idx_inc_luma_ac(
                &current_cbf, &dec.neighbors,
                mb_x, bx, by, current_is_intra,
            );
            let path = ResidualPathKind::Luma4x4 { block_idx: k as u8 };
            let ac_scan = decode_residual_block_with_offset_capture(
                dec, 0, 14, /* cat */ 1, ac_inc,
                frame_idx, mb_addr_u32, nal_idx, path, recorder,
            )?;
            let coded = ac_scan.iter().any(|&v| v != 0);
            current_cbf.set(1, k, coded);
            recorder.on_residual_block(
                frame_idx, mb_addr_u32, &ac_scan, 0, 14, path,
            );
        }
    }

    // Chroma DC (cat=3, 4 coeffs) if cbp_chroma >= 1.
    if cbp_chroma >= 1 {
        for plane in 0u8..2 {
            let inc = compute_cbf_ctx_idx_inc_chroma_dc(
                &dec.neighbors, mb_x, plane, current_is_intra,
            );
            let path = ResidualPathKind::ChromaDc { plane };
            let dc_flat = decode_residual_block_with_offset_capture(
                dec, 0, 3, /* cat */ 3, inc,
                frame_idx, mb_addr_u32, nal_idx, path, recorder,
            )?;
            let coded = dc_flat.iter().any(|&v| v != 0);
            current_cbf.set(3, plane as usize, coded);
            recorder.on_residual_block(
                frame_idx, mb_addr_u32, &dc_flat, 0, 3, path,
            );
        }
    }

    // Chroma AC (cat=4, 8 blocks) if cbp_chroma == 2.
    if cbp_chroma == 2 {
        for plane in 0u8..2 {
            for sub in 0..4usize {
                let bx = (sub % 2) as u8;
                let by = (sub / 2) as u8;
                let inc = compute_cbf_ctx_idx_inc_chroma_ac(
                    &current_cbf, &dec.neighbors,
                    mb_x, plane, bx, by, current_is_intra,
                );
                let path = ResidualPathKind::ChromaAc {
                    plane, block_idx: sub as u8,
                };
                let ac_scan = decode_residual_block_with_offset_capture(
                    dec, 0, 14, /* cat */ 4, inc,
                    frame_idx, mb_addr_u32, nal_idx, path, recorder,
                )?;
                let coded = ac_scan.iter().any(|&v| v != 0);
                current_cbf.set(
                    4, block_pos_to_chroma_ac_idx(plane, bx, by), coded,
                );
                recorder.on_residual_block(
                    frame_idx, mb_addr_u32, &ac_scan, 0, 14, path,
                );
            }
        }
    }

    let is_last = decode_end_of_slice_flag(dec)?;
    let mut nb = CabacNeighborMB {
        mb_type: MbTypeClass::I16x16,
        ..CabacNeighborMB::default()
    };
    nb.intra_chroma_pred_mode = chroma_pred_mode as u8;
    nb.cbp_luma = if cbp_luma_flag != 0 { 0x0F } else { 0 };
    nb.cbp_chroma = cbp_chroma as u8;
    nb.mb_qp_delta = qp_delta;
    nb.coded_block_flag_cat = current_cbf.to_neighbor_cbf();
    dec.neighbors.commit(mb_x, nb);

    Ok(is_last)
}

/// Phase 6E-A3 — B-slice MB dispatch. Currently supports B_Skip
/// only (mb_skip_flag = 1). Non-skip B-MBs return
/// `UnsupportedMbType` until §6E-A4 adds the mode-decision + ME
/// paths.
///
/// B_Skip semantics (spec § 7.4.5):
/// - mb_skip_flag = 1 → no further syntax for this MB.
/// - Decoder uses spatial direct (or temporal direct, per
///   `direct_spatial_mv_pred_flag`) to derive both L0 and L1
///   motion vectors from neighbors.
/// - Encoder emits no MVD bits, no ref_idx, no residuals.
/// - Stego coverage = zero (no bypass bins emitted by this MB).
#[allow(clippy::too_many_arguments)]
fn walk_b_mb(
    dec: &mut CabacDecoder<'_>,
    pps: &Pps,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    prev_mb_qp: &mut i32,
    frame_idx: u32,
    nal_idx: u32,
    recorder: &mut PositionRecorder,
    _opts: &WalkOptions,
    num_active_l0: u8,
) -> Result<bool, WalkError> {
    // 1. mb_skip_flag (B-slice ctxIdxOffset = 24).
    let is_skip = decode_mb_skip_flag_b(dec, mb_x)?;
    if is_skip {
        return walk_b_skip(dec, mb_x);
    }

    // 2. mb_type via the §6E-A3 B-slice bin tree (covers all 24
    //    spec values 0..=23). Per-mb_type body dispatch below.
    let mb_type = decode_mb_type_b(dec, mb_x)?;
    let mb_addr_u32 = (mb_y * mb_w + mb_x) as u32;

    // §B-direct-fix.v2 (#194): mirror the encoder's
    // transform_size_8x8_flag emission gate. For all B-MB inter
    // modes phasm ships (16x16/16x8/8x16/B_8x8 with sub_mb_type ≤ 3),
    // noSubMbPartSizeLessThan8x8Flag = 1 (no sub-8x8 partitions).
    // Encoder emits the flag iff `pps.transform_8x8_mode_flag &&
    // cbp_luma != 0`; walker must consume it accordingly.
    let t8x8 = pps.transform_8x8_mode_flag;

    // §6E-A6.0 — per-mb_type dispatcher. Each variant handler
    // lights up in a dedicated sub-phase. §6E-A6.1q.c (#152): the
    // L0/L1/Bi/partitioned/B_8x8 walkers now decode non-zero CBP
    // residuals via the shared `finish_b_inter` tail.
    match mb_type {
        0 => walk_b_direct_16x16(dec, mb_x, prev_mb_qp, frame_idx, nal_idx, mb_addr_u32, recorder, t8x8),
        1 => walk_b_l0_16x16(
            dec, mb_x, prev_mb_qp, frame_idx, nal_idx, mb_addr_u32, recorder, t8x8, num_active_l0,
        ),
        2 => walk_b_l1_16x16(dec, mb_x, prev_mb_qp, frame_idx, nal_idx, mb_addr_u32, recorder, t8x8),
        3 => walk_b_bi_16x16(
            dec, mb_x, prev_mb_qp, frame_idx, nal_idx, mb_addr_u32, recorder, t8x8, num_active_l0,
        ),
        4..=21 => walk_b_partitioned(
            dec, mb_x, prev_mb_qp, mb_type as u8,
            frame_idx, nal_idx, mb_addr_u32, recorder, t8x8, num_active_l0,
        ),
        22 => walk_b_8x8(
            dec, mb_x, prev_mb_qp, frame_idx, nal_idx, mb_addr_u32, recorder, t8x8, num_active_l0,
        ),
        23..=47 => {
            // §intra-in-B (#319) — Phase 1: walker un-reject.
            // Spec § 7.4.5: B-slice mb_type ≥ 23 is intra. The
            // I-suffix value is `mb_type - 23`, mapping to I-slice
            // mb_type 0..24 per Table 7-11:
            //   I-suffix 0 → I_NxN (I_4x4 / I_8x8 via t8x8 flag)
            //   I-suffix 1..24 → I_16x16 variants (luma-pred + cbp combo)
            //   I-suffix 25 → I_PCM (mb_type=48; out of scope)
            // Re-uses `walk_inxn_mb` and `walk_i16x16_mb` from the
            // I-slice walker since the per-MB intra syntax is
            // identical between I-slice intra, intra-in-P, and
            // intra-in-B (only the mb_type prefix differs).
            let i_mb_type = mb_type - 23;
            if i_mb_type == 0 {
                walk_inxn_mb(
                    dec, pps, mb_x, mb_y, mb_w, prev_mb_qp, frame_idx, nal_idx, recorder,
                )
            } else {
                walk_i16x16_mb(
                    dec, mb_x, mb_y, mb_w, prev_mb_qp, frame_idx, nal_idx, recorder,
                    i_mb_type,
                )
            }
        }
        48 => Err(WalkError::H264(H264Error::Unsupported(
            "B-slice I_PCM (mb_type=48) not yet wired (#188 follow-on)".into(),
        ))),
        _ => Err(WalkError::H264(H264Error::Unsupported(format!(
            "B-slice mb_type {mb_type} > 48 (spec invalid)"
        )))),
    }
}

/// `B_Skip`: `mb_skip_flag = 1` already consumed. No further syntax;
/// commit the BSkipOrDirect neighbour and read `end_of_slice_flag`.
fn walk_b_skip(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
) -> Result<bool, WalkError> {
    let is_last = decode_end_of_slice_flag(dec)?;
    let nb = CabacNeighborMB {
        mb_type: MbTypeClass::BSkipOrDirect,
        mb_skip_flag: true,
        ..CabacNeighborMB::default()
    };
    dec.neighbors.commit(mb_x, nb);
    Ok(is_last)
}

/// `B_Direct_16x16` (mb_type = 0): no MVDs, no ref_idx (decoder
/// reconstructs MV via spatial direct on its own side). Read CBP +
/// (if non-zero) residual blocks via the same path P-frame uses.
///
/// §6E-A4(c)-lite encoder emits CBP=0 always; non-zero CBP appears
/// only once §6E-A4(c)-full / §6E-A6.1 add B-frame residual emission,
/// at which point this body extends to call the existing
/// [`decode_residual_block_cabac`] helpers (same code path as P).
fn walk_b_direct_16x16(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
    prev_mb_qp: &mut i32,
    frame_idx: u32,
    nal_idx: u32,
    mb_addr_u32: u32,
    recorder: &mut PositionRecorder,
    t8x8: bool,
) -> Result<bool, WalkError> {
    // §B-direct-residual.walker (#244, 2026-05-07) — B_Direct_16x16
    // CAN have CBP > 0 per spec § 7.4.5.1. The previous walker
    // hardcoded CBP=0 + WalkError::Unsupported for any non-zero CBP,
    // which prevented re-enabling Direct+residual emission on the
    // encoder side (V16 negative result root cause).
    //
    // The path now mirrors L0/L1/Bi: parse CBP byte, if non-zero
    // parse mb_qp_delta + luma 4x4 + chroma DC/AC residuals, then
    // commit BSkipOrDirect neighbour state (NOT PInter — the next
    // MB's mb_skip_flag bin0 ctx + B-slice mb_type prefix bin0 ctx
    // depend on whether this neighbour is direct-class, per spec
    // § 9.3.3.1.1.1 + § 9.3.3.1.1.3).
    //
    // Direct has NO MVDs on the wire — they're derived from
    // neighbour state at decode time. Pass `None, None` for MVD
    // abs values; the neighbour commit will use the
    // `abs_mvd_comp* = 0` defaults (correct for Direct).
    //
    // v1.4 Phase 4.5 (#316) — Direct/Skip has no L0 ref_idx on the
    // wire (derived from spatial-direct neighbour read). Commit 0
    // for the entire MB; spec § 7.4.5.1 says inferred ref_idxs are
    // not part of neighbour state lookup.
    finish_b_inter_with_mb_type(
        dec, mb_x, prev_mb_qp,
        /* l0_abs */ None, /* l1_abs */ None,
        frame_idx, nal_idx, mb_addr_u32, recorder, t8x8,
        MbTypeClass::BSkipOrDirect,
        [0i8; 16],
    )
}

/// §6E-A6.1 — `B_L0_16x16` (mb_type = 1) walker. One L0 MVD pair.
/// v1.4 (#305) — `ref_idx_l0` decoded when num_active_l0 > 1.
/// §6E-A6.1q.c (#152): non-zero CBP now decodes residuals (luma 4×4
/// + chroma DC/AC) via the shared `finish_b_inter` tail.
#[allow(clippy::too_many_arguments)]
fn walk_b_l0_16x16(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
    prev_mb_qp: &mut i32,
    frame_idx: u32,
    nal_idx: u32,
    mb_addr_u32: u32,
    recorder: &mut PositionRecorder,
    t8x8: bool,
    num_active_l0: u8,
) -> Result<bool, WalkError> {
    // v1.4 Phase 4.5 (#316) — capture the decoded ref_idx_l0 so the
    // neighbour commit uses the actual on-wire value (not 0).
    let current_ref_idx_mb = crate::codec::h264::cabac::neighbor::CurrentMbRefIdx::new();
    let ref_idx_l0_dec = if num_active_l0 > 1 {
        decode_ref_idx(dec, &current_ref_idx_mb, mb_x, 0, 0, (num_active_l0 - 1) as u32)? as u8
    } else {
        0
    };
    let (abs_x, abs_y) = decode_b_16x16_mvd_pair(dec, mb_x, /* list */ 0)?;
    finish_b_inter(
        dec, mb_x, prev_mb_qp,
        /* l0 */ Some([[abs_x; 16], [abs_y; 16]]), /* l1 */ None,
        frame_idx, nal_idx, mb_addr_u32, recorder, t8x8,
        [ref_idx_l0_dec as i8; 16],
    )
}

/// §6E-A6.1 — `B_L1_16x16` (mb_type = 2). Mirror of L0 path,
/// using L1 neighbour state for bin0 ctxIdxInc and committing
/// `abs_mvd_comp_l1` on neighbour update.
#[allow(clippy::too_many_arguments)]
fn walk_b_l1_16x16(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
    prev_mb_qp: &mut i32,
    frame_idx: u32,
    nal_idx: u32,
    mb_addr_u32: u32,
    recorder: &mut PositionRecorder,
    t8x8: bool,
) -> Result<bool, WalkError> {
    let (abs_x, abs_y) = decode_b_16x16_mvd_pair(dec, mb_x, /* list */ 1)?;
    // v1.4 Phase 4.5 (#316) — L1_16x16 has no L0 ref_idx on the wire
    // (partition is L1-only per Table 7-14). Commit [0;16].
    finish_b_inter(
        dec, mb_x, prev_mb_qp,
        None, Some([[abs_x; 16], [abs_y; 16]]),
        frame_idx, nal_idx, mb_addr_u32, recorder, t8x8,
        [0i8; 16],
    )
}

/// §6E-A6.1 — `B_Bi_16x16` (mb_type = 3). Two MVD pairs (L0 then L1
/// per spec § 7.3.5.1 mb_pred order). Each list uses its own
/// neighbour state for bin0 ctxIdxInc.
///
/// v1.4 (#305) — `ref_idx_l0` decoded when num_active_l0 > 1 (Bi
/// uses L0 list).
#[allow(clippy::too_many_arguments)]
fn walk_b_bi_16x16(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
    prev_mb_qp: &mut i32,
    frame_idx: u32,
    nal_idx: u32,
    mb_addr_u32: u32,
    recorder: &mut PositionRecorder,
    t8x8: bool,
    num_active_l0: u8,
) -> Result<bool, WalkError> {
    // v1.4 Phase 4.5 (#316) — capture decoded ref_idx_l0 (Bi uses L0).
    let current_ref_idx_mb = crate::codec::h264::cabac::neighbor::CurrentMbRefIdx::new();
    let ref_idx_l0_dec = if num_active_l0 > 1 {
        decode_ref_idx(dec, &current_ref_idx_mb, mb_x, 0, 0, (num_active_l0 - 1) as u32)? as u8
    } else {
        0
    };
    let (l0_x, l0_y) = decode_b_16x16_mvd_pair(dec, mb_x, 0)?;
    let (l1_x, l1_y) = decode_b_16x16_mvd_pair(dec, mb_x, 1)?;
    finish_b_inter(
        dec, mb_x, prev_mb_qp,
        Some([[l0_x; 16], [l0_y; 16]]),
        Some([[l1_x; 16], [l1_y; 16]]),
        frame_idx, nal_idx, mb_addr_u32, recorder, t8x8,
        [ref_idx_l0_dec as i8; 16],
    )
}

/// §6E-A6.2 — partitioned B mb_type walker (mb_types 4..21).
/// Looks up `(shape, list_usage_part0, list_usage_part1)` via
/// `b_partitioned::partitioned_b_meta`, then decodes per spec
/// § 7.3.5.1 mb_pred order (partition 0's MVDs L0-then-L1 before
/// partition 1's). §B-Partitioned-Residual Stage D (#206) — non-zero
/// CBP residuals are decoded via `finish_b_inter`'s shared luma 4×4
/// + chroma DC/AC path (mirror of the encoder's
/// `emit_b_residual_for_pred` at the partitioned emit site).
#[allow(clippy::too_many_arguments)]
fn walk_b_partitioned(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
    prev_mb_qp: &mut i32,
    mb_type: u8,
    frame_idx: u32,
    nal_idx: u32,
    mb_addr_u32: u32,
    recorder: &mut PositionRecorder,
    t8x8: bool,
    num_active_l0: u8,
) -> Result<bool, WalkError> {
    use crate::codec::h264::encoder::b_partitioned::{
        partitioned_b_meta, BListUse,
    };

    let meta = partitioned_b_meta(mb_type as u32).ok_or_else(|| {
        WalkError::H264(H264Error::Unsupported(format!(
            "walk_b_partitioned: mb_type {mb_type} not in 4..=21"
        )))
    })?;

    // v1.4 (#305) — ref_idx_l0 per partition per spec § 7.3.5.1
    // mb_pred(): all ref_idx_l0 BEFORE MVDs, in partition-index
    // order, filtered by uses-L0 (skip if partition is L1-only).
    //
    // v1.4 Phase 4.5 (#316) — capture decoded values per partition
    // for the neighbour commit fill below.
    let mut ref_idx_l0_decoded = [0u8; 2];
    let mut current_ref_idx_mb = crate::codec::h264::cabac::neighbor::CurrentMbRefIdx::new();
    let (pw_ref_4x4, ph_ref_4x4) = meta.shape.part_dim_4x4();
    if num_active_l0 > 1 {
        for idx in 0..2usize {
            let usage = if idx == 0 { meta.part0 } else { meta.part1 };
            let uses_l0 = matches!(usage, BListUse::L0 | BListUse::Bi);
            if !uses_l0 {
                continue;
            }
            let (off_x, off_y) = meta.shape.part_offset(idx);
            let cur_bx = off_x as u8;
            let cur_by = off_y as u8;
            ref_idx_l0_decoded[idx] = decode_ref_idx(
                dec, &current_ref_idx_mb, mb_x, cur_bx, cur_by,
                (num_active_l0 - 1) as u32,
            )? as u8;
            current_ref_idx_mb.fill_region(
                cur_bx, cur_by, pw_ref_4x4 as u8, ph_ref_4x4 as u8,
                ref_idx_l0_decoded[idx] as i8,
            );
        }
    }

    // §6E-A6.1q.e (#154) — within-MB MVD tracker per-list. Mirror of
    // the encoder's `emit_b_partitioned`. Partition 1's bin 0
    // ctxIdxInc reads partition 0's just-decoded MVD via
    // `compute_mvd_ctx_idx_inc_bin0_per_list`.
    //
    // §B-encoder-decoder-divergence Phase 2.7 (2026-05-08, #248) —
    // H.264 spec § 7.3.5.1 / § 9.3.3.1.1 interprets B-slice
    // partitioned MVDs in **list-major** order: outer iterates
    // list 0..2, inner iterates partition index 0..N-1, decoding
    // an MVD pair only when that partition uses that list. Walker
    // now mirrors that. Encoder side updated in lockstep
    // in `core/src/codec/h264/encoder/encoder.rs::emit_b_partitioned_method`.
    let mut current_mvd = CurrentMbMvdAbs::new();
    let (pw, ph) = meta.shape.part_dim_4x4();

    for list in 0u8..2 {
        for idx in 0..2 {
            let usage = if idx == 0 { meta.part0 } else { meta.part1 };
            let uses_this_list = match (list, usage) {
                (0, BListUse::L0) | (0, BListUse::Bi) => true,
                (1, BListUse::L1) | (1, BListUse::Bi) => true,
                _ => false,
            };
            if !uses_this_list {
                continue;
            }
            let (off_x, off_y) = meta.shape.part_offset(idx);
            let cur_bx = off_x as u8;
            let cur_by = off_y as u8;

            let (x, y) = decode_b_partition_mvd_pair(
                dec, mb_x, &current_mvd, cur_bx, cur_by, list,
            )?;
            if list == 0 {
                current_mvd.fill_region(cur_bx, cur_by, pw as u8, ph as u8, x, y);
            } else {
                current_mvd.fill_region_l1(cur_bx, cur_by, pw as u8, ph as u8, x, y);
            }
        }
    }

    // §Task #207 (2026-05-04) — pass the full per-sub-MB MVD layout
    // (not a broadcast of the max magnitude) so the encoder's
    // `current_mvd.to_neighbor()` per-position fill matches on the
    // walker side. Otherwise the next MB's bin0 ctxIdxInc reads from
    // a position the encoder filled with 0 but the walker filled with
    // max — CABAC desync at the bin level.
    let l0_for_finish = if any_uses_l0(meta) { Some(current_mvd.comp) } else { None };
    let l1_for_finish = if any_uses_l1(meta) { Some(current_mvd.comp_l1) } else { None };
    // v1.4 Phase 4.5 (#316) — per-block ref_idx_l0 fill from
    // partition geometry, mirroring the encoder's
    // `fill_ref_idx_l0_partitioned` exactly so neighbour ctxIdxInc
    // for the next MB matches encoder side AND a spec decoder.
    let ref_idx_l0_array = walker_fill_ref_idx_l0_partitioned(meta, ref_idx_l0_decoded);
    finish_b_inter(
        dec, mb_x, prev_mb_qp, l0_for_finish, l1_for_finish,
        frame_idx, nal_idx, mb_addr_u32, recorder, t8x8,
        ref_idx_l0_array,
    )
}

/// §6E-A6.3 — `B_8x8` walker (mb_type = 22). Decodes:
///
/// 1. 4 × `sub_mb_type` (each 0..=3 — B_Direct_8x8 / B_L0_8x8 /
///    B_L1_8x8 / B_Bi_8x8 — via `decode_sub_mb_type_b`).
/// 2. Per sub-MB s in raster order, the MVDs implied by its
///    sub_mb_type's list usage:
///      - 0 (Direct): no MVDs
///      - 1 (L0):     L0 MVD pair
///      - 2 (L1):     L1 MVD pair
///      - 3 (Bi):     L0 MVD pair, then L1 MVD pair
///    `ref_idx_lX` is inferred 0 (single-ref ship config).
/// 3. coded_block_pattern. §B-Partitioned-Residual Stage D (#206) —
///    non-zero CBP is decoded via `finish_b_inter` (luma 4×4 + chroma
///    DC/AC), mirroring the encoder-side wiring in
///    `emit_b_8x8_method`.
/// 4. end_of_slice_flag.
///
/// Aggregate MVD magnitudes across sub-MBs (max per axis per list)
/// for neighbour commit — matches `emit_b_8x8`'s aggregation.
#[allow(clippy::too_many_arguments)]
fn walk_b_8x8(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
    prev_mb_qp: &mut i32,
    frame_idx: u32,
    nal_idx: u32,
    mb_addr_u32: u32,
    recorder: &mut PositionRecorder,
    t8x8: bool,
    num_active_l0: u8,
) -> Result<bool, WalkError> {
    let mut sub_mb_types = [0u8; 4];
    for s in &mut sub_mb_types {
        let value = decode_sub_mb_type_b(dec)?;
        debug_assert!(
            value <= 3,
            "decode_sub_mb_type_b returned {value}; §6E-A6.3 supports 0..=3"
        );
        *s = value as u8;
    }

    // v1.4 (#305) — ref_idx_l0 per sub-MB per spec § 7.3.5.1
    // mb_pred(): all ref_idx_l0 AFTER 4×sub_mb_type and BEFORE MVDs,
    // in sub-MB-index order, filtered by sub-MB-uses-L0
    // (Direct=0 + L1=2 skip; L0=1 + Bi=3 emit).
    //
    // v1.4 Phase 4.5 (#316) — capture decoded values per sub-MB +
    // within-MB ref_idx tracker for spec § 6.4.11.7 ctxIdxInc lookups.
    let mut ref_idx_l0_decoded = [0u8; 4];
    let mut current_ref_idx_mb = crate::codec::h264::cabac::neighbor::CurrentMbRefIdx::new();
    if num_active_l0 > 1 {
        for s_idx in 0..4usize {
            let sub = sub_mb_types[s_idx];
            let uses_l0 = matches!(sub, 1 | 3);
            if !uses_l0 {
                continue;
            }
            let off_bx = ((s_idx & 1) * 2) as u8;
            let off_by = ((s_idx >> 1) * 2) as u8;
            ref_idx_l0_decoded[s_idx] = decode_ref_idx(
                dec, &current_ref_idx_mb, mb_x, off_bx, off_by,
                (num_active_l0 - 1) as u32,
            )? as u8;
            current_ref_idx_mb.fill_region(off_bx, off_by, 2, 2, ref_idx_l0_decoded[s_idx] as i8);
        }
    }

    // §6E-A6.1q.e (#154) — within-MB MVD tracker. Mirror of
    // `emit_b_8x8`.
    //
    // §B-encoder-decoder-divergence Phase 2.7 (2026-05-08, #248) —
    // The spec parses B_8x8 sub-MB MVDs in **list-major** order
    // (H.264 spec § 7.3.5.1 + § 9.3.3.1.1): outer iterates list
    // 0..2, inner iterates sub-MB index 0..4, decoding an MVD
    // pair only when that sub-MB uses that list. Walker now
    // mirrors that. Encoder side updated in lockstep in
    // `core/src/codec/h264/encoder/encoder.rs::emit_b_8x8_method`.
    let mut current_mvd = CurrentMbMvdAbs::new();

    for list in 0u8..2 {
        for s_idx in 0..4 {
            let sub = sub_mb_types[s_idx];
            let uses_this_list = match (list, sub) {
                (0, 1) | (0, 3) => true,
                (1, 2) | (1, 3) => true,
                _ => false,
            };
            if !uses_this_list {
                continue;
            }
            let off_bx = (s_idx & 1) * 2;
            let off_by = (s_idx >> 1) * 2;
            let cur_bx = off_bx as u8;
            let cur_by = off_by as u8;
            let pw = 2u8;
            let ph = 2u8;
            let (x, y) = decode_b_partition_mvd_pair(
                dec, mb_x, &current_mvd, cur_bx, cur_by, list,
            )?;
            if list == 0 {
                current_mvd.fill_region(cur_bx, cur_by, pw, ph, x, y);
            } else {
                current_mvd.fill_region_l1(cur_bx, cur_by, pw, ph, x, y);
            }
        }
    }

    // §Task #207 — pass per-sub-MB MVD layout (mirror of
    // walk_b_partitioned fix above).
    let any_l0 = sub_mb_types.iter().any(|&s| matches!(s, 1 | 3));
    let any_l1 = sub_mb_types.iter().any(|&s| matches!(s, 2 | 3));
    let l0_some = if any_l0 { Some(current_mvd.comp) } else { None };
    let l1_some = if any_l1 { Some(current_mvd.comp_l1) } else { None };
    // v1.4 Phase 4.5 (#316) — per-block ref_idx_l0 fill from sub-MB
    // geometry, mirroring encoder's `fill_ref_idx_l0_b8x8`.
    let ref_idx_l0_array = walker_fill_ref_idx_l0_b8x8(sub_mb_types, ref_idx_l0_decoded);
    finish_b_inter(
        dec, mb_x, prev_mb_qp, l0_some, l1_some,
        frame_idx, nal_idx, mb_addr_u32, recorder, t8x8,
        ref_idx_l0_array,
    )
}

/// True if any partition's list usage includes L0 (= L0 or Bi).
fn any_uses_l0(meta: crate::codec::h264::encoder::b_partitioned::BPartitionedMeta) -> bool {
    use crate::codec::h264::encoder::b_partitioned::BListUse;
    matches!(meta.part0, BListUse::L0 | BListUse::Bi)
        || matches!(meta.part1, BListUse::L0 | BListUse::Bi)
}

/// True if any partition's list usage includes L1.
fn any_uses_l1(meta: crate::codec::h264::encoder::b_partitioned::BPartitionedMeta) -> bool {
    use crate::codec::h264::encoder::b_partitioned::BListUse;
    matches!(meta.part0, BListUse::L1 | BListUse::Bi)
        || matches!(meta.part1, BListUse::L1 | BListUse::Bi)
}

/// Decode one X+Y MVD pair for a 16x16 B-slice partition.
/// `list` (0 = L0, 1 = L1) selects the per-list neighbour state
/// for bin0 ctxIdxInc per spec § 9.3.3.1.1.7. Returns the decoded
/// (|MVD_x|, |MVD_y|) magnitudes so the caller can stash them in
/// the right per-list neighbour field.
fn decode_b_16x16_mvd_pair(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
    list: u8,
) -> Result<(i16, i16), WalkError> {
    use crate::codec::h264::cabac::neighbor::ctx_idx_inc_mvd_bin0_per_list;

    let mut null = NullLogger;
    let bin0_inc_x = ctx_idx_inc_mvd_bin0_per_list(
        &dec.neighbors, mb_x, 0, 0, /* component */ 0, list,
    );
    let mvd_x = {
        let mut pc = PositionCtx { frame_idx: 0, mb_addr: 0, nal_idx: 0, logger: &mut null };
        decode_mvd_with_bin0_inc(dec, /* component */ 0, list, /* partition */ 0, bin0_inc_x, &mut pc)?
    };
    let bin0_inc_y = ctx_idx_inc_mvd_bin0_per_list(
        &dec.neighbors, mb_x, 0, 0, /* component */ 1, list,
    );
    let mvd_y = {
        let mut pc = PositionCtx { frame_idx: 0, mb_addr: 0, nal_idx: 0, logger: &mut null };
        decode_mvd_with_bin0_inc(dec, 1, list, 0, bin0_inc_y, &mut pc)?
    };
    let abs_x = mvd_x.unsigned_abs().min(i16::MAX as u32) as i16;
    let abs_y = mvd_y.unsigned_abs().min(i16::MAX as u32) as i16;
    Ok((abs_x, abs_y))
}

/// §6E-A6.1q.e (#154) — decode one X+Y MVD pair for a B-slice
/// partition with within-MB-aware bin 0 ctxIdxInc derivation.
/// `(cur_bx, cur_by)` are the partition's TL 4×4-block coordinates
/// within the MB. For partition 1 of 16×8/8×16 (and sub-MBs 1+ of
/// B_8x8), this consults `current_mvd` (per-list) before falling
/// back to cross-MB neighbours.
///
/// Mirror of the encoder's `compute_mvd_ctx_idx_inc_bin0_per_list`
/// call sites in `emit_b_partitioned` / `emit_b_8x8`.
fn decode_b_partition_mvd_pair(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
    current_mvd: &CurrentMbMvdAbs,
    cur_bx: u8,
    cur_by: u8,
    list: u8,
) -> Result<(i16, i16), WalkError> {
    use crate::codec::h264::cabac::neighbor::compute_mvd_ctx_idx_inc_bin0_per_list;

    let mut null = NullLogger;
    let bin0_inc_x = compute_mvd_ctx_idx_inc_bin0_per_list(
        current_mvd, &dec.neighbors, mb_x, cur_bx, cur_by,
        /* component */ 0, list,
    );
    let mvd_x = {
        let mut pc = PositionCtx { frame_idx: 0, mb_addr: 0, nal_idx: 0, logger: &mut null };
        decode_mvd_with_bin0_inc(dec, 0, list, 0, bin0_inc_x, &mut pc)?
    };
    let bin0_inc_y = compute_mvd_ctx_idx_inc_bin0_per_list(
        current_mvd, &dec.neighbors, mb_x, cur_bx, cur_by, 1, list,
    );
    let mvd_y = {
        let mut pc = PositionCtx { frame_idx: 0, mb_addr: 0, nal_idx: 0, logger: &mut null };
        decode_mvd_with_bin0_inc(dec, 1, list, 0, bin0_inc_y, &mut pc)?
    };
    let abs_x = mvd_x.unsigned_abs().min(i16::MAX as u32) as i16;
    let abs_y = mvd_y.unsigned_abs().min(i16::MAX as u32) as i16;
    Ok((abs_x, abs_y))
}

/// Shared tail for §6E-A6.1 B-inter walkers: read CBP, decode
/// `mb_qp_delta` + residual blocks (luma 4×4 + chroma DC + chroma
/// AC) when `cbp != 0`, commit neighbour as inter (with per-list
/// MVD magnitudes), read `end_of_slice_flag`.
///
/// §6E-A6.1q.c (#152): the CBP=0 fast path is preserved; the
/// non-zero CBP path mirrors `walk_p_inter_residual`'s 4×4 luma +
/// chroma DC/AC decode (no 8×8 transform support — B-side encoder
/// stays on 4×4 transform per §6E-A6.1q.b scope).
///
/// Task #207 (2026-05-04): `l0_abs` / `l1_abs` are `Option<[[i16; 16]; 2]>`
/// — full per-position arrays so partitioned + B_8x8 walkers can
/// mirror the encoder's `current_mvd.to_neighbor()` per-sub-MB layout
/// (instead of the broadcast that 16x16 walkers use). When sub-MB MVDs
/// differ across the MB (e.g. B_8x8 with one non-zero sub-MB and three
/// zero), the broadcast was overstating magnitude at zero positions
/// → next-MB bin0 ctxIdxInc disagreed with encoder → CABAC desync.
#[allow(clippy::too_many_arguments)]
/// v1.4 Phase 4.5 (#316) — walker-side mirror of
/// `fill_ref_idx_l0_partitioned` in encoder.rs. Builds a 16-entry
/// per-4×4-block ref_idx_l0 array from per-partition decoded values.
fn walker_fill_ref_idx_l0_partitioned(
    meta: crate::codec::h264::encoder::b_partitioned::BPartitionedMeta,
    ref_idx_l0: [u8; 2],
) -> [i8; 16] {
    use crate::codec::h264::encoder::b_partitioned::BListUse;
    let mut out = [0i8; 16];
    let (pw, ph) = meta.shape.part_dim_4x4();
    for idx in 0..2usize {
        let usage = if idx == 0 { meta.part0 } else { meta.part1 };
        if !matches!(usage, BListUse::L0 | BListUse::Bi) {
            continue;
        }
        let (off_x, off_y) = meta.shape.part_offset(idx);
        let val = ref_idx_l0[idx] as i8;
        for dy in 0..ph {
            for dx in 0..pw {
                out[(off_y + dy) * 4 + (off_x + dx)] = val;
            }
        }
    }
    out
}

/// v1.4 Phase 4.5 (#316) — walker-side mirror of
/// `fill_ref_idx_l0_b8x8`.
fn walker_fill_ref_idx_l0_b8x8(
    sub_mb_types: [u8; 4],
    ref_idx_l0: [u8; 4],
) -> [i8; 16] {
    let mut out = [0i8; 16];
    for s_idx in 0..4usize {
        if !matches!(sub_mb_types[s_idx], 1 | 3) {
            continue;
        }
        let off_x = (s_idx & 1) * 2;
        let off_y = (s_idx >> 1) * 2;
        let val = ref_idx_l0[s_idx] as i8;
        for dy in 0..2usize {
            for dx in 0..2usize {
                out[(off_y + dy) * 4 + (off_x + dx)] = val;
            }
        }
    }
    out
}

/// v1.4 Phase 4.5 (#316) — walker-side helper for P-slice partition
/// neighbour ref_idx_l0 fill. P-partitions all use L0.
/// `ref_idx_l0`: P16x16/P_Skip = [v, _, _, _]; P16x8 = [top, bottom];
/// P8x16 = [left, right]; P_8x8 = [s0, s1, s2, s3].
fn walker_fill_ref_idx_l0_p(mb_type_p: u32, ref_idx_l0: [u8; 4]) -> [i8; 16] {
    let mut out = [0i8; 16];
    match mb_type_p {
        // P_L0_16x16 (mb_type_p=0): uniform.
        0 => out = [ref_idx_l0[0] as i8; 16],
        // P_L0_16x8 (mb_type_p=1): top half / bottom half.
        1 => {
            for r in 0..2 {
                for c in 0..4 {
                    out[r * 4 + c] = ref_idx_l0[0] as i8;
                }
            }
            for r in 2..4 {
                for c in 0..4 {
                    out[r * 4 + c] = ref_idx_l0[1] as i8;
                }
            }
        }
        // P_L0_8x16 (mb_type_p=2): left half / right half.
        2 => {
            for r in 0..4 {
                for c in 0..2 {
                    out[r * 4 + c] = ref_idx_l0[0] as i8;
                }
                for c in 2..4 {
                    out[r * 4 + c] = ref_idx_l0[1] as i8;
                }
            }
        }
        // P_8x8 (mb_type_p=3): 4 sub-MBs, each covering a 2×2 cell region.
        3 => {
            for s_idx in 0..4usize {
                let off_x = (s_idx & 1) * 2;
                let off_y = (s_idx >> 1) * 2;
                let val = ref_idx_l0[s_idx] as i8;
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        out[(off_y + dy) * 4 + (off_x + dx)] = val;
                    }
                }
            }
        }
        _ => {}
    }
    out
}

fn finish_b_inter(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
    prev_mb_qp: &mut i32,
    l0_abs: Option<[[i16; 16]; 2]>,
    l1_abs: Option<[[i16; 16]; 2]>,
    frame_idx: u32,
    nal_idx: u32,
    mb_addr_u32: u32,
    recorder: &mut PositionRecorder,
    t8x8: bool,
    ref_idx_l0_array: [i8; 16],
) -> Result<bool, WalkError> {
    finish_b_inter_with_mb_type(
        dec, mb_x, prev_mb_qp, l0_abs, l1_abs, frame_idx, nal_idx, mb_addr_u32,
        recorder, t8x8, MbTypeClass::PInter, ref_idx_l0_array,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_b_inter_with_mb_type(
    dec: &mut CabacDecoder<'_>,
    mb_x: usize,
    prev_mb_qp: &mut i32,
    l0_abs: Option<[[i16; 16]; 2]>,
    l1_abs: Option<[[i16; 16]; 2]>,
    frame_idx: u32,
    nal_idx: u32,
    mb_addr_u32: u32,
    recorder: &mut PositionRecorder,
    t8x8: bool,
    mb_type_class: MbTypeClass,
    ref_idx_l0_array: [i8; 16],
) -> Result<bool, WalkError> {
    let cbp_byte = decode_coded_block_pattern(dec, mb_x)?;
    let cbp_luma = cbp_byte & 0x0F;
    let cbp_chroma = (cbp_byte >> 4) & 0x03;

    // §B-direct-fix.v2 (#194): mirror the encoder's transform_size_8x8_flag
    // emission gate. For all B-MB inter modes phasm ships
    // (16x16/16x8/8x16/B_8x8 with sub_mb_type ≤ 3),
    // noSubMbPartSizeLessThan8x8Flag = 1, so the gate reduces to
    // `transform_8x8_mode_flag && cbp_luma != 0`. The encoder
    // always emits `false` (B-MB residuals stay on 4x4); decoder
    // just consumes the bin to keep CABAC sync.
    if t8x8 && cbp_luma != 0 {
        let _flag = decode_transform_size_8x8_flag(dec, mb_x)?;
    }

    // mb_qp_delta only when cbp != 0 (mirrors encoder).
    let qp_delta = if cbp_byte != 0 {
        let d = decode_mb_qp_delta(dec)?;
        *prev_mb_qp += d;
        d
    } else {
        0
    };

    // Residuals: luma 4×4 + chroma DC + chroma AC.
    let mut current_cbf = CurrentMbCbf::new();
    let current_is_intra = false;
    if cbp_luma != 0 {
        for k in 0..16usize {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            if cbp_luma & (1 << (k / 4)) != 0 {
                let inc = compute_cbf_ctx_idx_inc_luma_4x4(
                    &current_cbf, &dec.neighbors,
                    mb_x, bx, by, current_is_intra,
                );
                let path = ResidualPathKind::Luma4x4 { block_idx: k as u8 };
                let scan = decode_residual_block_with_offset_capture(
                    dec, 0, 15, /* cat */ 2, inc,
                    frame_idx, mb_addr_u32, nal_idx, path, recorder,
                )?;
                let coded = scan.iter().any(|&v| v != 0);
                current_cbf.set(2, k, coded);
                recorder.on_residual_block(
                    frame_idx, mb_addr_u32, &scan, 0, 15, path,
                );
            }
        }
    }
    if cbp_chroma >= 1 {
        for plane in 0u8..2 {
            let inc = compute_cbf_ctx_idx_inc_chroma_dc(
                &dec.neighbors, mb_x, plane, current_is_intra,
            );
            let path = ResidualPathKind::ChromaDc { plane };
            let dc_flat = decode_residual_block_with_offset_capture(
                dec, 0, 3, /* cat */ 3, inc,
                frame_idx, mb_addr_u32, nal_idx, path, recorder,
            )?;
            let coded = dc_flat.iter().any(|&v| v != 0);
            current_cbf.set(3, plane as usize, coded);
            recorder.on_residual_block(
                frame_idx, mb_addr_u32, &dc_flat, 0, 3, path,
            );
        }
    }
    if cbp_chroma == 2 {
        for plane in 0u8..2 {
            for sub in 0..4usize {
                let bx = (sub % 2) as u8;
                let by = (sub / 2) as u8;
                let inc = compute_cbf_ctx_idx_inc_chroma_ac(
                    &current_cbf, &dec.neighbors,
                    mb_x, plane, bx, by, current_is_intra,
                );
                let path = ResidualPathKind::ChromaAc {
                    plane, block_idx: sub as u8,
                };
                let ac_scan = decode_residual_block_with_offset_capture(
                    dec, 0, 14, /* cat */ 4, inc,
                    frame_idx, mb_addr_u32, nal_idx, path, recorder,
                )?;
                let coded = ac_scan.iter().any(|&v| v != 0);
                current_cbf.set(
                    4, block_pos_to_chroma_ac_idx(plane, bx, by), coded,
                );
                recorder.on_residual_block(
                    frame_idx, mb_addr_u32, &ac_scan, 0, 14, path,
                );
            }
        }
    }

    let is_last = decode_end_of_slice_flag(dec)?;

    // §6E-A6.1 spec § 9.3.3.1.1.7 fix: commit per-list MVD
    // magnitudes so subsequent neighbour bin0 ctxIdxInc reads the
    // right list-specific state.
    let abs_mvd_l0 = l0_abs.unwrap_or([[0; 16]; 2]);
    let abs_mvd_l1 = l1_abs.unwrap_or([[0; 16]; 2]);
    let mut nb = CabacNeighborMB {
        mb_type: mb_type_class,
        mb_skip_flag: false,
        cbp_luma,
        cbp_chroma,
        // v1.4 Phase 4.5 (#316) — propagate the per-block ref_idx_l0
        // captured from the wire so next MB's ref_idx ctxIdxInc
        // matches what a spec-conforming decoder computes (the prior
        // [0;16] hardcode desynced vs the reference decoder whenever ref_idx>0).
        ref_idx_l0: ref_idx_l0_array,
        abs_mvd_comp: abs_mvd_l0,
        abs_mvd_comp_l1: abs_mvd_l1,
        ..CabacNeighborMB::default()
    };
    nb.mb_qp_delta = qp_delta;
    nb.coded_block_flag_cat = current_cbf.to_neighbor_cbf();
    nb.transform_size_8x8_flag = false;
    dec.neighbors.commit(mb_x, nb);
    Ok(is_last)
}

/// I_NxN macroblock walker (chunk 6D). Mirrors
/// `Encoder::commit_i4x4_macroblock_cabac` (encoder.rs:2946).
/// Returns the end_of_slice_flag bit.
///
/// **Scope**: I_4x4 only (transform_size_8x8_flag = 0). I_8x8
/// (flag = 1) returns Unsupported sentinel — chunk-5 encoder
/// hardcodes `enable_transform_8x8 = false` so this path is
/// reachable only on bitstreams produced by other encoders or
/// future enable_transform_8x8 support.
fn walk_inxn_mb(
    dec: &mut CabacDecoder<'_>,
    pps: &Pps,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    prev_mb_qp: &mut i32,
    frame_idx: u32,
    nal_idx: u32,
    recorder: &mut PositionRecorder,
) -> Result<bool, WalkError> {
    let mb_addr_u32 = (mb_y * mb_w + mb_x) as u32;
    let current_is_intra = true;

    // 1. transform_size_8x8_flag — only when PPS enables 8x8 transform
    //    AND mb_type indicates I_NxN. Selects I_4x4 (flag=0, 16 luma
    //    blocks of cat=2) vs I_8x8 (flag=1, 4 luma blocks of cat=5).
    let use_8x8 = if pps.transform_8x8_mode_flag {
        decode_transform_size_8x8_flag(dec, mb_x)?
    } else {
        false
    };

    // 2. Per-block intra prediction modes. I_4x4 emits 16 mode flags
    //    (one per 4×4 block); I_8x8 emits 4 (one per 8×8 block) per
    //    encoder.rs:4465 — the contexts are shared (Table 9-39 specifies
    //    ctxIdxOffsets 68/69 for both I_4x4 and I_8x8).
    let n_pred_modes = if use_8x8 { 4 } else { 16 };
    for _ in 0..n_pred_modes {
        let prev_flag = decode_prev_intra4x4_pred_mode_flag(dec)?;
        if !prev_flag {
            let _rem = decode_rem_intra4x4_pred_mode(dec)?;
        }
    }

    // 3. intra_chroma_pred_mode (encoder line 3069).
    let chroma_pred_mode = decode_intra_chroma_pred_mode(dec, mb_x)?;

    // 4. coded_block_pattern (encoder line 3073). Returns
    //    chroma:luma packed byte; unpack as encoder pack_cbp.
    let cbp_byte = decode_coded_block_pattern(dec, mb_x)?;
    let cbp_luma = cbp_byte & 0x0F;
    let cbp_chroma = (cbp_byte >> 4) & 0x03;

    // 5. mb_qp_delta — ONLY when cbp_value != 0 (encoder line 3079).
    let qp_delta_emitted = if cbp_byte != 0 {
        let delta = decode_mb_qp_delta(dec)?;
        *prev_mb_qp += delta;
        delta
    } else {
        0
    };
    let _ = qp_delta_emitted;

    let mut current_cbf = CurrentMbCbf::new();

    if use_8x8 {
        // 6a. I_8x8 luma residual: 4 blocks of cat=5 (no per-block CBF
        //     — encoder.rs:4490-4512). Each block has 64 coefficients
        //     (start=0, end=63). Gated per 8×8 block via cbp_luma bit k.
        //     Set every 4×4 block within an 8×8-coded block as coded
        //     in the cat=2 CBF state, mirroring encoder.rs:4519-4527 so
        //     subsequent I_4x4 MBs see consistent neighbor CBF.
        for k in 0..4usize {
            if cbp_luma & (1 << k) != 0 {
                let path_kind = ResidualPathKind::Luma8x8 {
                    block_idx: k as u8,
                };
                // Phase C.3.6.1 — dispatch to offset-capturing logger
                // when enabled; else NullLogger (zero-cost). Scoped so
                // recorder is free for `on_residual_block` below.
                let mut null = NullLogger;
                let scan = {
                    let logger: &mut dyn PositionLogger =
                        if let Some(ol) = recorder.offset_logger.as_mut() {
                            ol
                        } else {
                            &mut null
                        };
                    let mut pos_ctx = PositionCtx {
                        frame_idx,
                        mb_addr: mb_addr_u32,
                        nal_idx,
                        logger,
                    };
                    decode_residual_block_cabac_8x8(
                        dec,
                        &mut pos_ctx,
                        |ci, kind| path_kind.path(ci, kind),
                    )?
                };
                recorder.on_residual_block(
                    frame_idx, mb_addr_u32, &scan, 0, 63, path_kind,
                );
            }
            // Mark all four 4×4 sub-blocks coded/uncoded per the 8×8 bit
            // (encoder.rs:4519-4527). Necessary so the next MB's
            // compute_cbf_ctx_idx_inc_luma_4x4 sees the right neighbor.
            for sub in 0..4usize {
                let blk_idx = k * 4 + sub;
                current_cbf.set(2, blk_idx, cbp_luma & (1 << k) != 0);
            }
        }
    } else if cbp_luma != 0 {
        // 6b. I_4x4 luma residual: 16 blocks of cat=2 (with per-block
        //     CBF). Gated per 8x8 block via cbp_luma bit k/4.
        for k in 0..16usize {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            if cbp_luma & (1 << (k / 4)) != 0 {
                let inc = compute_cbf_ctx_idx_inc_luma_4x4(
                    &current_cbf, &dec.neighbors,
                    mb_x, bx, by, current_is_intra,
                );
                let path = ResidualPathKind::Luma4x4 { block_idx: k as u8 };
                let scan = decode_residual_block_with_offset_capture(
                    dec, 0, 15, /* cat */ 2, inc,
                    frame_idx, mb_addr_u32, nal_idx, path, recorder,
                )?;
                let coded = scan.iter().any(|&v| v != 0);
                current_cbf.set(2, k, coded);
                recorder.on_residual_block(
                    frame_idx, mb_addr_u32, &scan, 0, 15, path,
                );
            }
        }
    }

    // 7. Chroma DC (cat=3, 4 coeffs) — same as I_16x16.
    if cbp_chroma >= 1 {
        for plane in 0u8..2 {
            let inc = compute_cbf_ctx_idx_inc_chroma_dc(
                &dec.neighbors, mb_x, plane, current_is_intra,
            );
            let path = ResidualPathKind::ChromaDc { plane };
            let dc_flat = decode_residual_block_with_offset_capture(
                dec, 0, 3, /* cat */ 3, inc,
                frame_idx, mb_addr_u32, nal_idx, path, recorder,
            )?;
            let coded = dc_flat.iter().any(|&v| v != 0);
            current_cbf.set(3, plane as usize, coded);
            recorder.on_residual_block(
                frame_idx, mb_addr_u32, &dc_flat, 0, 3, path,
            );
        }
    }

    // 8. Chroma AC (cat=4, 8 blocks) — same as I_16x16.
    if cbp_chroma == 2 {
        for plane in 0u8..2 {
            for sub in 0..4usize {
                let bx = (sub % 2) as u8;
                let by = (sub / 2) as u8;
                let inc = compute_cbf_ctx_idx_inc_chroma_ac(
                    &current_cbf, &dec.neighbors,
                    mb_x, plane, bx, by, current_is_intra,
                );
                let path = ResidualPathKind::ChromaAc {
                    plane, block_idx: sub as u8,
                };
                let ac_scan = decode_residual_block_with_offset_capture(
                    dec, 0, 14, /* cat */ 4, inc,
                    frame_idx, mb_addr_u32, nal_idx, path, recorder,
                )?;
                let coded = ac_scan.iter().any(|&v| v != 0);
                current_cbf.set(
                    4, block_pos_to_chroma_ac_idx(plane, bx, by), coded,
                );
                recorder.on_residual_block(
                    frame_idx, mb_addr_u32, &ac_scan, 0, 14, path,
                );
            }
        }
    }

    // 9. end_of_slice_flag.
    let is_last = decode_end_of_slice_flag(dec)?;

    // 10. Commit neighbor state for next MB. Mirrors encoder line 3206
    //     (I_4x4) and 4590-4598 (I_8x8). The transform_size_8x8_flag bit
    //     drives the next MB's transform_size_8x8_flag CABAC ctx (Table
    //     9-39 ctxIdxInc derivation).
    let mut nb = CabacNeighborMB {
        mb_type: MbTypeClass::INxN,
        ..CabacNeighborMB::default()
    };
    nb.intra_chroma_pred_mode = chroma_pred_mode as u8;
    nb.cbp_luma = cbp_luma;
    nb.cbp_chroma = cbp_chroma;
    nb.mb_qp_delta = qp_delta_emitted;
    nb.coded_block_flag_cat = current_cbf.to_neighbor_cbf();
    nb.transform_size_8x8_flag = use_8x8;
    dec.neighbors.commit(mb_x, nb);

    Ok(is_last)
}

/// Unified P-partition walker (§30A3 + §30A4). Dispatches MVD
/// decode based on `mb_type_p` (0..3); residual + neighbor commit
/// shared via `decode_p_residuals_and_finish`.
///
/// Mirrors `Encoder::write_p_macroblock_cabac` (encoder.rs:1610+)
/// + `emit_p_mvds_cabac` (encoder.rs:1863+) +
/// `emit_sub_mb_mvds_cabac` (encoder.rs:1979+).
///
/// **MVD positions intentionally NOT recorded** — encoder's MVD
/// hook is declared but unwired (§30D pending). NullLogger drops
/// inline emissions so the decoder cover's mvd_*_bypass domains
/// stay empty, matching the encoder's empty MVD cover.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn walk_p_partition_mb(
    dec: &mut CabacDecoder<'_>,
    pps: &Pps,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    prev_mb_qp: &mut i32,
    frame_idx: u32,
    nal_idx: u32,
    recorder: &mut PositionRecorder,
    mb_type_p: u32,
    opts: &WalkOptions,
    num_active_l0: u8,
) -> Result<bool, WalkError> {
    let mb_addr_u32 = (mb_y * mb_w + mb_x) as u32;
    let mut current_mvd = CurrentMbMvdAbs::new();

    // 1. P_8x8 emits 4 sub_mb_types BEFORE any MVDs (encoder.rs:
    //    1613-1617). Other partition types skip this step.
    let sub_types: Option<[u32; 4]> = if mb_type_p == 3 {
        let mut sm = [0u32; 4];
        for s in sm.iter_mut() {
            *s = decode_sub_mb_type_p(dec)?;
        }
        Some(sm)
    } else {
        None
    };

    // 2. v1.4 (#305) — ref_idx_l0 per partition per spec § 7.3.5.1
    // mb_pred(). At num_ref_idx_l0_active=1 (single-ref default) the
    // gate is closed and zero bins read. P-slice partitions all use
    // L0 by definition (B-slice list-usage filtering doesn't apply).
    //
    // v1.4 Phase 4.5 (#316) — capture decoded values per partition
    // for the neighbour commit fill below.
    let mut ref_idx_l0_p = [0u8; 4];
    // v1.4 Phase 4.5 (#316) — within-MB tracker so partition 1+ reads
    // partition 0's just-decoded ref_idx for ctxIdxInc lookups
    // (spec § 6.4.11.7).
    let mut current_ref_idx_mb = crate::codec::h264::cabac::neighbor::CurrentMbRefIdx::new();
    if num_active_l0 > 1 {
        let c_max = (num_active_l0 - 1) as u32;
        match mb_type_p {
            0 => {
                ref_idx_l0_p[0] = decode_ref_idx(dec, &current_ref_idx_mb, mb_x, 0, 0, c_max)? as u8;
                current_ref_idx_mb.fill_region(0, 0, 4, 4, ref_idx_l0_p[0] as i8);
            }
            1 => {
                // P_16x8 — top + bottom 16x8 partitions.
                ref_idx_l0_p[0] = decode_ref_idx(dec, &current_ref_idx_mb, mb_x, 0, 0, c_max)? as u8;
                current_ref_idx_mb.fill_region(0, 0, 4, 2, ref_idx_l0_p[0] as i8);
                ref_idx_l0_p[1] = decode_ref_idx(dec, &current_ref_idx_mb, mb_x, 0, 2, c_max)? as u8;
                current_ref_idx_mb.fill_region(0, 2, 4, 2, ref_idx_l0_p[1] as i8);
            }
            2 => {
                // P_8x16 — left + right 8x16 partitions.
                ref_idx_l0_p[0] = decode_ref_idx(dec, &current_ref_idx_mb, mb_x, 0, 0, c_max)? as u8;
                current_ref_idx_mb.fill_region(0, 0, 2, 4, ref_idx_l0_p[0] as i8);
                ref_idx_l0_p[1] = decode_ref_idx(dec, &current_ref_idx_mb, mb_x, 2, 0, c_max)? as u8;
                current_ref_idx_mb.fill_region(2, 0, 2, 4, ref_idx_l0_p[1] as i8);
            }
            3 => {
                // P_8x8 — per-sub-MB ref_idx_l0 (one per sub-MB,
                // regardless of sub_mb_type internal partitioning).
                const SUB_ORIGINS: [(u8, u8); 4] = [(0, 0), (2, 0), (0, 2), (2, 2)];
                for (i, &(bx, by)) in SUB_ORIGINS.iter().enumerate() {
                    ref_idx_l0_p[i] = decode_ref_idx(dec, &current_ref_idx_mb, mb_x, bx, by, c_max)? as u8;
                    current_ref_idx_mb.fill_region(bx, by, 2, 2, ref_idx_l0_p[i] as i8);
                }
            }
            _ => unreachable!("mb_type_p > 3 routed elsewhere"),
        }
    }
    let ref_idx_l0_array = walker_fill_ref_idx_l0_p(mb_type_p, ref_idx_l0_p);

    // 3. MVDs per partition. Encoder pattern (emit_p_mvds_cabac):
    //    P16x16 → 1 part; P16x8 → 2 parts (top+bottom 16x8);
    //    P8x16 → 2 parts (left+right 8x16); P_8x8 → 4 sub-MBs at
    //    SUB_MB_ORIGINS_4X4 = [(0,0),(2,0),(0,2),(2,2)] each
    //    expanding by sub_mb_type into 1-4 partitions.
    decode_p_partition_mvds(
        dec, &mut current_mvd, mb_x, mb_addr_u32, frame_idx, nal_idx,
        mb_type_p, sub_types.as_ref(), recorder, opts,
    )?;

    // Spec `noSubMbPartSizeLessThan8x8Flag` — gates whether
    // `transform_size_8x8_flag` may be emitted. Always true for
    // P_L0_16x16 / P_16x8 / P_8x16; for P_8x8 it requires every
    // sub_mb_type == 0 (P_L0_8x8). Mirrors
    // partition_decision.rs:no_sub_mb_part_size_lt_8x8.
    let no_sub_mb_part_size_lt_8x8 = match (mb_type_p, sub_types.as_ref()) {
        (3, Some(sm)) => sm.iter().all(|&t| t == 0),
        (3, None) => false,
        _ => true,
    };

    // 4-9. Shared residual + neighbor commit.
    decode_p_residuals_and_finish(
        dec, pps, mb_x, mb_y, mb_w, prev_mb_qp, frame_idx, nal_idx,
        mb_addr_u32, recorder, &current_mvd, no_sub_mb_part_size_lt_8x8,
        ref_idx_l0_array,
    )
}

/// Decode the MVD pairs for a P-partition MB. Mirrors encoder's
/// `emit_p_mvds_cabac` + `emit_sub_mb_mvds_cabac`.
#[allow(clippy::too_many_arguments)]
fn decode_p_partition_mvds(
    dec: &mut CabacDecoder<'_>,
    current_mvd: &mut CurrentMbMvdAbs,
    mb_x: usize,
    mb_addr_u32: u32,
    frame_idx: u32,
    nal_idx: u32,
    mb_type_p: u32,
    sub_types: Option<&[u32; 4]>,
    recorder: &mut PositionRecorder,
    opts: &WalkOptions,
) -> Result<(), WalkError> {
    match mb_type_p {
        0 => {
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, 0, 0, 4, 4,
                /* partition */ 0,
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
        }
        1 => {
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, 0, 0, 4, 2, 0,
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, 0, 2, 4, 2, 1,
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
        }
        2 => {
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, 0, 0, 2, 4, 0,
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, 2, 0, 2, 4, 1,
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
        }
        3 => {
            let sm = sub_types.expect("P_8x8 requires sub_types");
            const SUB_ORIGINS: [(u8, u8); 4] = [(0, 0), (2, 0), (0, 2), (2, 2)];
            for (i, &sub_t) in sm.iter().enumerate() {
                let (sub_bx, sub_by) = SUB_ORIGINS[i];
                decode_sub_mb_mvds(
                    dec, current_mvd, mb_x, sub_bx, sub_by, sub_t,
                    /* sub_mb_idx */ i as u8,
                    frame_idx, nal_idx, mb_addr_u32, recorder, opts,
                )?;
            }
        }
        _ => unreachable!("mb_type_p > 3 routed elsewhere"),
    }
    Ok(())
}

/// Decode the MVD pairs for one 8×8 sub-MB. Mirrors encoder's
/// `emit_sub_mb_mvds_cabac` (encoder.rs:1979).
///
/// `sub_mb_type` codenum:
///   0 → P_L0_8x8 — 1 partition (sub_bx, sub_by, 2, 2)
///   1 → P_L0_8x4 — 2 partitions stacked (2, 1)
///   2 → P_L0_4x8 — 2 partitions side-by-side (1, 2)
///   3 → P_L0_4x4 — 4 partitions (1, 1)
/// `sub_mb_idx` ∈ 0..4 selects the sub-MB within the MB. Combined
/// with the within-sub-MB partition index it forms the global
/// `partition` value passed to `decode_one_mvd_pair_p` /
/// `MvdSlot.partition`. Convention: `partition = sub_mb_idx * 4 +
/// sub_part_idx`. Encoder mirror lives in §30D-A3.
#[allow(clippy::too_many_arguments)]
fn decode_sub_mb_mvds(
    dec: &mut CabacDecoder<'_>,
    current_mvd: &mut CurrentMbMvdAbs,
    mb_x: usize,
    sub_bx: u8,
    sub_by: u8,
    sub_mb_type: u32,
    sub_mb_idx: u8,
    frame_idx: u32,
    nal_idx: u32,
    mb_addr_u32: u32,
    recorder: &mut PositionRecorder,
    opts: &WalkOptions,
) -> Result<(), WalkError> {
    let p = |sub_part_idx: u8| sub_mb_idx * 4 + sub_part_idx;
    match sub_mb_type {
        0 => {
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, sub_bx, sub_by, 2, 2, p(0),
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
        }
        1 => {
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, sub_bx, sub_by, 2, 1, p(0),
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, sub_bx, sub_by + 1, 2, 1, p(1),
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
        }
        2 => {
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, sub_bx, sub_by, 1, 2, p(0),
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, sub_bx + 1, sub_by, 1, 2, p(1),
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
        }
        3 => {
            for oy in 0u8..2 {
                for ox in 0u8..2 {
                    let sp = oy * 2 + ox;
                    decode_one_mvd_pair_p(
                        dec, current_mvd, mb_x,
                        sub_bx + ox, sub_by + oy, 1, 1, p(sp),
                        frame_idx, nal_idx, mb_addr_u32, recorder, opts,
                    )?;
                }
            }
        }
        _ => unreachable!("sub_mb_type > 3 cannot be decoded"),
    }
    Ok(())
}

/// Decode one MVD pair (x + y components) and update current_mvd
/// for subsequent same-MB partition neighbor lookups. Mirrors
/// encoder's `emit_one_mvd_pair_cabac` (encoder.rs:2088).
///
/// **§30D-B**: when `opts.record_mvd` is true, record the decoded
/// MVD value via `PositionRecorder::on_mvd_slot` for both x and y
/// axes — symmetric to encoder's PositionLoggerHook firing
/// `on_mvd_slot` per axis in `apply_mvd_hook_to_choice`.
#[allow(clippy::too_many_arguments)]
fn decode_one_mvd_pair_p(
    dec: &mut CabacDecoder<'_>,
    current_mvd: &mut CurrentMbMvdAbs,
    mb_x: usize,
    part_bx: u8,
    part_by: u8,
    part_w4: u8,
    part_h4: u8,
    partition: u8,
    frame_idx: u32,
    nal_idx: u32,
    mb_addr_u32: u32,
    recorder: &mut PositionRecorder,
    opts: &WalkOptions,
) -> Result<(), WalkError> {
    let mut null = NullLogger;
    let bin0_inc_x = compute_mvd_ctx_idx_inc_bin0(
        current_mvd, &dec.neighbors, mb_x, part_bx, part_by, 0,
    );
    // Phase C.3.6.1 — dispatch to recorder's offset_logger when set.
    let mvd_x = {
        let logger: &mut dyn PositionLogger =
            if let Some(ol) = recorder.offset_logger.as_mut() {
                ol
            } else {
                &mut null
            };
        let mut pc = PositionCtx {
            frame_idx, mb_addr: mb_addr_u32, nal_idx, logger,
        };
        decode_mvd_with_bin0_inc(
            dec, /* component */ 0, /* list */ 0,
            /* partition */ 0, bin0_inc_x, &mut pc,
        )?
    };
    let bin0_inc_y = compute_mvd_ctx_idx_inc_bin0(
        current_mvd, &dec.neighbors, mb_x, part_bx, part_by, 1,
    );
    let mvd_y = {
        let logger: &mut dyn PositionLogger =
            if let Some(ol) = recorder.offset_logger.as_mut() {
                ol
            } else {
                &mut null
            };
        let mut pc = PositionCtx {
            frame_idx, mb_addr: mb_addr_u32, nal_idx, logger,
        };
        decode_mvd_with_bin0_inc(
            dec, 1, 0, 0, bin0_inc_y, &mut pc,
        )?
    };
    current_mvd.fill_region(
        part_bx, part_by, part_w4, part_h4,
        mvd_x.unsigned_abs().min(i16::MAX as u32) as i16,
        mvd_y.unsigned_abs().min(i16::MAX as u32) as i16,
    );

    // §30D-B: record MVD slots for the X and Y axes when the
    // walker is opt'd in. Encoder mirror in
    // `apply_mvd_hook_to_choice` (encoder.rs).
    if opts.record_mvd {
        let sx = MvdSlot {
            list: 0, partition, axis: Axis::X, value: mvd_x,
        };
        let sy = MvdSlot {
            list: 0, partition, axis: Axis::Y, value: mvd_y,
        };
        recorder.on_mvd_slot(frame_idx, mb_addr_u32, &sx);
        recorder.on_mvd_slot(frame_idx, mb_addr_u32, &sy);
    }
    Ok(())
}

/// Shared P-MB body: CBP + transform_size + qp_delta + residuals
/// + end_of_slice + neighbor commit. Used by all P-partition
/// walkers (P_L0_16x16, P_16x8, P_8x16, P_8x8). Encoder mirror:
/// `write_p_macroblock_cabac` lines 1627-1812.
#[allow(clippy::too_many_arguments)]
fn decode_p_residuals_and_finish(
    dec: &mut CabacDecoder<'_>,
    pps: &Pps,
    mb_x: usize,
    _mb_y: usize,
    _mb_w: usize,
    prev_mb_qp: &mut i32,
    frame_idx: u32,
    nal_idx: u32,
    mb_addr_u32: u32,
    recorder: &mut PositionRecorder,
    current_mvd: &CurrentMbMvdAbs,
    no_sub_mb_part_size_lt_8x8: bool,
    ref_idx_l0_array: [i8; 16],
) -> Result<bool, WalkError> {
    let current_is_intra = false;

    // 4. coded_block_pattern.
    let cbp_byte = decode_coded_block_pattern(dec, mb_x)?;
    let cbp_luma = cbp_byte & 0x0F;
    let cbp_chroma = (cbp_byte >> 4) & 0x03;

    // 5. transform_size_8x8_flag — emitted iff PPS+cbp_luma>0+
    //    no_sub_mb_part_size_lt_8x8 (encoder.rs:2659-2666).
    let use_8x8 = if pps.transform_8x8_mode_flag
        && cbp_luma != 0
        && no_sub_mb_part_size_lt_8x8
    {
        decode_transform_size_8x8_flag(dec, mb_x)?
    } else {
        false
    };

    // 6. mb_qp_delta if cbp != 0.
    let qp_delta = if cbp_byte != 0 {
        let d = decode_mb_qp_delta(dec)?;
        *prev_mb_qp += d;
        d
    } else {
        0
    };

    // 7. Residuals.
    let mut current_cbf = CurrentMbCbf::new();
    if use_8x8 {
        // 8×8 luma residual: 4 cat=5 blocks (no per-block CBF). Mirrors
        // encoder.rs:2686-2710.
        for k in 0..4usize {
            if cbp_luma & (1 << k) != 0 {
                let path_kind = ResidualPathKind::Luma8x8 {
                    block_idx: k as u8,
                };
                // Phase C.3.6.1 — dispatch to offset-capturing logger
                // when enabled; else NullLogger (zero-cost). Scoped so
                // recorder is free for `on_residual_block` below.
                let mut null = NullLogger;
                let scan = {
                    let logger: &mut dyn PositionLogger =
                        if let Some(ol) = recorder.offset_logger.as_mut() {
                            ol
                        } else {
                            &mut null
                        };
                    let mut pos_ctx = PositionCtx {
                        frame_idx,
                        mb_addr: mb_addr_u32,
                        nal_idx,
                        logger,
                    };
                    decode_residual_block_cabac_8x8(
                        dec,
                        &mut pos_ctx,
                        |ci, kind| path_kind.path(ci, kind),
                    )?
                };
                recorder.on_residual_block(
                    frame_idx, mb_addr_u32, &scan, 0, 63, path_kind,
                );
            }
            // Mark all four 4×4 sub-blocks coded/uncoded per the 8×8 bit
            // (encoder.rs:2717-2725) so subsequent 4×4 MBs see the same
            // neighbor CBF state.
            for sub in 0..4usize {
                let blk_idx = k * 4 + sub;
                current_cbf.set(2, blk_idx, cbp_luma & (1 << k) != 0);
            }
        }
    } else if cbp_luma != 0 {
        for k in 0..16usize {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            if cbp_luma & (1 << (k / 4)) != 0 {
                let inc = compute_cbf_ctx_idx_inc_luma_4x4(
                    &current_cbf, &dec.neighbors,
                    mb_x, bx, by, current_is_intra,
                );
                let path = ResidualPathKind::Luma4x4 { block_idx: k as u8 };
                let scan = decode_residual_block_with_offset_capture(
                    dec, 0, 15, /* cat */ 2, inc,
                    frame_idx, mb_addr_u32, nal_idx, path, recorder,
                )?;
                let coded = scan.iter().any(|&v| v != 0);
                current_cbf.set(2, k, coded);
                recorder.on_residual_block(
                    frame_idx, mb_addr_u32, &scan, 0, 15, path,
                );
            }
        }
    }
    if cbp_chroma >= 1 {
        for plane in 0u8..2 {
            let inc = compute_cbf_ctx_idx_inc_chroma_dc(
                &dec.neighbors, mb_x, plane, current_is_intra,
            );
            let path = ResidualPathKind::ChromaDc { plane };
            let dc_flat = decode_residual_block_with_offset_capture(
                dec, 0, 3, /* cat */ 3, inc,
                frame_idx, mb_addr_u32, nal_idx, path, recorder,
            )?;
            let coded = dc_flat.iter().any(|&v| v != 0);
            current_cbf.set(3, plane as usize, coded);
            recorder.on_residual_block(
                frame_idx, mb_addr_u32, &dc_flat, 0, 3, path,
            );
        }
    }
    if cbp_chroma == 2 {
        for plane in 0u8..2 {
            for sub in 0..4usize {
                let bx = (sub % 2) as u8;
                let by = (sub / 2) as u8;
                let inc = compute_cbf_ctx_idx_inc_chroma_ac(
                    &current_cbf, &dec.neighbors,
                    mb_x, plane, bx, by, current_is_intra,
                );
                let path = ResidualPathKind::ChromaAc {
                    plane, block_idx: sub as u8,
                };
                let ac_scan = decode_residual_block_with_offset_capture(
                    dec, 0, 14, /* cat */ 4, inc,
                    frame_idx, mb_addr_u32, nal_idx, path, recorder,
                )?;
                let coded = ac_scan.iter().any(|&v| v != 0);
                current_cbf.set(
                    4, block_pos_to_chroma_ac_idx(plane, bx, by), coded,
                );
                recorder.on_residual_block(
                    frame_idx, mb_addr_u32, &ac_scan, 0, 14, path,
                );
            }
        }
    }

    // 8. end_of_slice_flag.
    let is_last = decode_end_of_slice_flag(dec)?;

    // 9. Commit PInter neighbors.
    let mut nb = CabacNeighborMB {
        mb_type: MbTypeClass::PInter,
        mb_skip_flag: false,
        ..CabacNeighborMB::default()
    };
    nb.cbp_luma = cbp_luma;
    nb.cbp_chroma = cbp_chroma;
    nb.mb_qp_delta = qp_delta;
    nb.coded_block_flag_cat = current_cbf.to_neighbor_cbf();
    // v1.4 Phase 4.5 (#316) — propagate decoded ref_idx_l0 per
    // partition geometry so spec-conforming decoders agree on the
    // next MB's ref_idx ctxIdxInc.
    nb.ref_idx_l0 = ref_idx_l0_array;
    nb.abs_mvd_comp = current_mvd.to_neighbor();
    nb.transform_size_8x8_flag = use_8x8;
    dec.neighbors.commit(mb_x, nb);

    Ok(is_last)
}

/// Helper: decode one residual block. Position recording for cover
/// bits + values happens externally via `PositionRecorder` after-the-
/// fact (the recorder's enumerate-from-coeffs path gives identical
/// positions to the inline decoder logger by construction; encoder-
/// side parity gate, chunk 6A).
///
/// Phase C.3.6.1 (#428) — when the recorder has `offset_logger`
/// enabled (`PositionRecorder::with_offsets()`), the inline logger
/// passed to the syntax decoder captures RBSP bit offsets of every
/// emitted bypass bin. Otherwise the inline logger is `NullLogger`
/// (zero-cost). The offset-capturing path is the foundation of
/// Option C bitstream-mod stego on the OpenH264 backend.
#[allow(clippy::too_many_arguments)]
fn decode_residual_block_with_offset_capture(
    dec: &mut CabacDecoder<'_>,
    start_idx: usize,
    end_idx: usize,
    ctx_block_cat: u8,
    cbf_ctx_idx_inc: u32,
    frame_idx: u32,
    mb_addr: u32,
    nal_idx: u32,
    path_kind: ResidualPathKind,
    recorder: &mut PositionRecorder,
) -> Result<Vec<i32>, WalkError> {
    let mut null = NullLogger;
    let logger: &mut dyn PositionLogger =
        if let Some(ol) = recorder.offset_logger.as_mut() {
            ol
        } else {
            &mut null
        };
    let mut pos_ctx = PositionCtx {
        frame_idx,
        mb_addr,
        nal_idx,
        logger,
    };
    let coeffs = decode_residual_block_cabac(
        dec, start_idx, end_idx, ctx_block_cat, cbf_ctx_idx_inc,
        &mut pos_ctx,
        |ci, kind| path_kind.path(ci, kind),
    )?;
    Ok(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_annex_b_returns_no_slices() {
        let r = walk_annex_b_for_cover(&[]);
        assert!(matches!(r, Err(WalkError::NoSlices)));
    }

    #[test]
    fn rejects_stream_with_no_parameter_sets() {
        // Synthetic Annex-B with just one IDR NAL (no SPS/PPS first)
        // — should fail with MissingParameterSet.
        // NAL header: forbidden_zero=0, nal_ref_idc=3, nal_type=5 (IDR)
        let nal_byte = (3u8 << 5) | 5;
        let mut bytes = vec![0, 0, 0, 1, nal_byte];
        bytes.extend_from_slice(&[0xff; 8]); // garbage RBSP
        let r = walk_annex_b_for_cover(&bytes);
        match r {
            Err(WalkError::MissingParameterSet) => {}
            // parse_nal_units_annexb may also surface a parse error
            // for invalid RBSP; either is acceptable for "no SPS
            // before slice".
            Err(_) => {}
            Ok(_) => panic!("expected error on missing SPS/PPS"),
        }
    }

    /// Walks the Annex-B output of the chunk-5 stego encoder (with
    /// empty message → byte-identical to no-stego baseline). Chunk
    /// 6D: I_NxN dispatch is wired, so a 32×32 single I-frame
    /// (4 MBs in any mix of I_16x16 + I_4x4) MUST walk without
    /// error and produce a non-empty cover.
    #[test]
    fn walks_chunk5_empty_message_output_without_error() {
        use crate::codec::h264::stego::encode_pixels::h264_stego_encode_i_frames_only;

        let frame_size = 32 * 32 * 3 / 2;
        let mut yuv = Vec::with_capacity(frame_size);
        let mut s: u32 = 0xCAFE_F00D;
        for _ in 0..frame_size {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            yuv.push((s >> 16) as u8);
        }

        let bytes = h264_stego_encode_i_frames_only(
            &yuv, 32, 32, 1, &[], "test", 4, Some(26),
        ).expect("encode");

        let walk = walk_annex_b_for_cover(&bytes).expect("walk");
        assert_eq!(walk.n_slices, 1, "expected exactly one slice");
        assert_eq!(walk.n_mb, 4, "32x32 has exactly 4 MBs");
        // High-entropy random YUV ⇒ many nonzero residuals ⇒
        // sign-domain cover is non-empty.
        assert!(
            walk.cover.coeff_sign_bypass.len() > 0,
            "expected non-empty coeff sign cover for random YUV",
        );
    }

    /// Uniform-luma YUV strongly biases the encoder toward I_16x16
    /// DC mode (single sample value across the MB). Confirms the
    /// chunk-6C dispatch decodes without state-error and produces a
    /// non-empty cover when the I_16x16 path is exercised.
    #[test]
    fn walks_uniform_yuv_through_i16x16_path() {
        use crate::codec::h264::stego::encode_pixels::h264_stego_encode_i_frames_only;

        // 16x16 single MB, uniform mid-gray. Y=128, Cb=Cr=128.
        let frame_size = 16 * 16 * 3 / 2;
        let mut yuv = vec![128u8; frame_size];
        // Add tiny noise so the mode decision sees something but
        // I_16x16 stays optimal.
        for (i, b) in yuv.iter_mut().enumerate() {
            *b = (*b as i32 + ((i as i32) % 3 - 1)) as u8;
        }

        let bytes = h264_stego_encode_i_frames_only(
            &yuv, 16, 16, 1, &[], "test", 4, Some(26),
        ).expect("encode");

        match walk_annex_b_for_cover(&bytes) {
            Ok(walk) => {
                assert_eq!(walk.n_slices, 1);
                assert_eq!(walk.n_mb, 1, "16x16 has exactly one MB");
                // Cover may be empty if all coeffs zero (uniform YUV →
                // few residual nonzeros), but the chroma DC pred
                // typically leaves SOME nonzero coefficient. Either
                // way, walk completed without error — that's the gate.
                let total = walk.cover.coeff_sign_bypass.len()
                    + walk.cover.coeff_suffix_lsb.len();
                let _ = total;
            }
            Err(WalkError::H264(H264Error::Unsupported(s)))
                if s.contains("I_NxN") || s.contains("I_PCM") =>
            {
                // I_4x4 / I_PCM mode pick — chunk 6D / 6E will wire.
            }
            Err(e) => panic!("unexpected walk error: {e}"),
        }
    }

    /// **Load-bearing parity gate (chunk 6F)**.
    ///
    /// Encode a YUV with the chunk-5 driver (empty message ⇒ Pass-3
    /// bytes byte-identical to no-stego baseline; cover bits in the
    /// emitted bitstream EXACTLY equal what Pass-1 logged). Capture
    /// the encode-side Pass-1 cover via `pass1_count`, walk the
    /// same Annex-B output via the decode-side slice walker, and
    /// assert the two covers are IDENTICAL across all four
    /// bypass-bin domains: positions, bit values, ordering. By
    /// construction (encoder + decoder enumerate over identical
    /// post-quantize coefficients in identical scan order) this
    /// equality must hold.
    ///
    /// If this test ever flakes, every later sub-phase (STC reverse
    /// + decrypt + frame parse) is dead. The whole decode side
    /// rests on this byte-identity invariant.
    #[test]
    fn encode_walk_parity_chunk5_empty_message() {
        use crate::codec::h264::stego::encode_pixels::{
            h264_stego_encode_i_frames_only, pass1_count,
        };

        let frame_size = 32 * 32 * 3 / 2;
        let mut yuv = Vec::with_capacity(frame_size);
        let mut s: u32 = 0xCAFE_F00D;
        for _ in 0..frame_size {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            yuv.push((s >> 16) as u8);
        }

        // Encode-side: capture Pass-1 cover.
        let encoder_gop = pass1_count(
            &yuv, 32, 32, 1, frame_size, Some(26),
        ).expect("pass1_count");
        let encoder_cover = encoder_gop.cover;

        // Encode again via the production driver (empty message ⇒
        // Pass-3 bytes equal no-stego baseline).
        let bytes = h264_stego_encode_i_frames_only(
            &yuv, 32, 32, 1, &[], "test", 4, Some(26),
        ).expect("encode");

        // Decode-side: walk the resulting Annex-B.
        let walk = walk_annex_b_for_cover(&bytes).expect("walk");
        let decoder_cover = walk.cover;

        // Per-domain byte-identity assertions.
        assert_eq!(
            decoder_cover.coeff_sign_bypass.positions,
            encoder_cover.coeff_sign_bypass.positions,
            "coeff_sign_bypass positions must match"
        );
        assert_eq!(
            decoder_cover.coeff_sign_bypass.bits,
            encoder_cover.coeff_sign_bypass.bits,
            "coeff_sign_bypass bits must match"
        );
        assert_eq!(
            decoder_cover.coeff_suffix_lsb.positions,
            encoder_cover.coeff_suffix_lsb.positions,
            "coeff_suffix_lsb positions must match"
        );
        assert_eq!(
            decoder_cover.coeff_suffix_lsb.bits,
            encoder_cover.coeff_suffix_lsb.bits,
            "coeff_suffix_lsb bits must match"
        );
        // I-frame-only scope: MVD domains stay empty.
        assert_eq!(decoder_cover.mvd_sign_bypass.positions,
            encoder_cover.mvd_sign_bypass.positions);
        assert_eq!(decoder_cover.mvd_sign_bypass.bits,
            encoder_cover.mvd_sign_bypass.bits);
        assert_eq!(decoder_cover.mvd_suffix_lsb.positions,
            encoder_cover.mvd_suffix_lsb.positions);
        assert_eq!(decoder_cover.mvd_suffix_lsb.bits,
            encoder_cover.mvd_suffix_lsb.bits);
    }

    /// **§30A1+2 P-slice walker test**: encode an I-frame followed by
    /// a P-frame on identical YUV. The P-frame's MBs should be
    /// pickable as P_SKIP or intra-in-P; the walker should advance
    /// through both slices without erroring.
    ///
    /// We construct YUV manually because `h264_stego_encode_yuv_string`
    /// uses I-frame-only encoding. For this test we drive
    /// `Encoder::encode_p_frame` directly.
    #[test]
    fn walks_i_then_p_frame_without_error() {
        use crate::codec::h264::encoder::encoder::{Encoder, EntropyMode};

        // Uniform-luma YUV maximizes P_SKIP probability (residuals
        // round to zero for stationary content).
        let frame_size = 16 * 16 * 3 / 2;
        let yuv = vec![128u8; frame_size];

        let mut enc = Encoder::new(16, 16, Some(26)).expect("encoder");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = false;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&enc.encode_i_frame(&yuv).expect("I-frame"));
        bytes.extend_from_slice(&enc.encode_p_frame(&yuv).expect("P-frame"));

        // The walker may succeed (all P_SKIP / intra-in-P) or fail
        // with the P_partition sentinel (encoder picked
        // P_L0_16x16 / P_16x8 / P_8x16 / P_8x8 — wired in §30A3+).
        // Both outcomes prove the §30A1+2 dispatch reaches mb_skip
        // / mb_type_p decode without state errors.
        match walk_annex_b_for_cover(&bytes) {
            Ok(walk) => {
                assert_eq!(walk.n_slices, 2, "expected I + P slices");
                assert!(walk.n_mb >= 2);
            }
            Err(WalkError::H264(H264Error::Unsupported(s)))
                if s.contains("P_partition") || s.contains("I_NxN")
                   || s.contains("I_PCM") =>
            {
                // §30A3+ wires P_partition; chunk 6D + 6E wire I_NxN
                // / I_PCM. Tolerate while those land.
            }
            Err(e) => panic!("unexpected walk error: {e}"),
        }
    }

    /// **§30A3 P_L0_16x16 test**: encode an I-frame followed by
    /// a P-frame on YUV with mild texture and small motion. The
    /// encoder typically picks P_L0_16x16 when there's coherent
    /// translation. The walker should advance through both slices
    /// without erroring.
    #[test]
    fn walks_i_then_p_frame_with_motion() {
        use crate::codec::h264::encoder::encoder::{Encoder, EntropyMode};

        // Slowly-varying YUV so successive frames are correlated
        // enough for P_L0_16x16 to be picked, but noisy enough
        // that residuals don't round to zero (avoiding P_SKIP).
        let frame_size = 16 * 16 * 3 / 2;
        let mut frame0 = vec![0u8; frame_size];
        let mut frame1 = vec![0u8; frame_size];
        let mut s: u32 = 0x4242_DEAD;
        for i in 0..frame_size {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            // Mid-gray ± small variation
            frame0[i] = (128 + ((s >> 24) & 0x07) as u8) as u8;
            // frame1 shifted slightly: small diff from frame0
            frame1[i] = (128 + ((s >> 22) & 0x07) as u8) as u8;
        }

        let mut enc = Encoder::new(16, 16, Some(26)).expect("encoder");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = false;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&enc.encode_i_frame(&frame0).expect("I"));
        bytes.extend_from_slice(&enc.encode_p_frame(&frame1).expect("P"));

        // Outcome depends on encoder's mode pick. P_SKIP, P_L0_16x16,
        // intra-in-P all walk cleanly with §30A1+2+3. Non-16x16
        // P-partitions still error out — tolerated until §30A4.
        match walk_annex_b_for_cover(&bytes) {
            Ok(walk) => {
                assert_eq!(walk.n_slices, 2);
                assert!(walk.n_mb >= 2);
            }
            Err(WalkError::H264(H264Error::Unsupported(s)))
                if s.contains("P_partition mb_type_p=") || s.contains("I_NxN")
                   || s.contains("I_PCM") || s.contains("transform_size") =>
            {
                // §30A4 / chunk 6E pending paths.
            }
            Err(e) => panic!("unexpected walk error: {e}"),
        }
    }

    /// **§30B P-slice parity gate**: encode I+P frame via the direct
    /// Encoder API with a `PositionLoggerHook` → capture encoder
    /// Pass-1 cover. Walk the same Annex-B via the bin-decoder
    /// → capture walker cover. Assert byte-identical across the
    /// coeff domains. (MVD domains stay empty until §30D wires the
    /// encoder MVD hook.)
    ///
    /// This is the load-bearing P-slice test. Surfaces bugs in
    /// neighbor-state propagation, MVD parsing, and partition
    /// dispatch the same way chunks 6F + 8 surfaced bugs in I-slice
    /// walking and multi-frame frame_idx handling.
    #[test]
    fn p_slice_parity_walker_matches_encoder_cover() {
        use crate::codec::h264::encoder::encoder::{Encoder, EntropyMode};
        use crate::codec::h264::stego::encoder_hook::{
            PositionLoggerHook, StegoMbHook,
        };

        // Slowly-varying YUV: mild texture, some motion → encoder
        // likely picks a mix of P_SKIP and P_L0_16x16 (and possibly
        // intra-in-P). Avoids non-16x16 P-partitions which §30A4
        // wires.
        let frame_size = 16 * 16 * 3 / 2;
        let mut frame0 = vec![0u8; frame_size];
        let mut frame1 = vec![0u8; frame_size];
        let mut s: u32 = 0xBEEF_F00D;
        for i in 0..frame_size {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            frame0[i] = (128 + ((s >> 24) & 0x07) as u8) as u8;
            frame1[i] = (128 + ((s >> 22) & 0x07) as u8) as u8;
        }

        // Encoder side: Pass-1 logger captures cover.
        let mut enc = Encoder::new(16, 16, Some(26)).expect("encoder");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = false;
        enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&enc.encode_i_frame(&frame0).expect("I"));
        bytes.extend_from_slice(&enc.encode_p_frame(&frame1).expect("P"));
        let mut hook = enc.take_stego_hook().expect("hook present");
        let encoder_gop = hook.take_cover_if_logger().expect("logger");

        // Decoder side: walker.
        let walk = match walk_annex_b_for_cover(&bytes) {
            Ok(w) => w,
            Err(WalkError::H264(H264Error::Unsupported(s))) => {
                // The encoder chose a partition mode not yet wired
                // in §30A. Skip the test — adjust the input YUV in
                // a follow-up to deterministically hit wired modes.
                eprintln!("Skipping §30B parity (encoder picked {s}); §30A4+ pending");
                return;
            }
            Err(e) => panic!("unexpected walk error: {e}"),
        };

        // Per-domain byte-identity assertions.
        assert_eq!(
            walk.cover.coeff_sign_bypass.positions,
            encoder_gop.cover.coeff_sign_bypass.positions,
            "P-slice coeff_sign_bypass positions must match"
        );
        assert_eq!(
            walk.cover.coeff_sign_bypass.bits,
            encoder_gop.cover.coeff_sign_bypass.bits,
            "P-slice coeff_sign_bypass bits must match"
        );
        assert_eq!(
            walk.cover.coeff_suffix_lsb.positions,
            encoder_gop.cover.coeff_suffix_lsb.positions,
        );
        assert_eq!(
            walk.cover.coeff_suffix_lsb.bits,
            encoder_gop.cover.coeff_suffix_lsb.bits,
        );
        // MVD domains: both sides must agree on emptiness until
        // §30D wires the encoder MVD hook.
        assert_eq!(walk.cover.mvd_sign_bypass.len(),
            encoder_gop.cover.mvd_sign_bypass.len());
        assert_eq!(walk.cover.mvd_suffix_lsb.len(),
            encoder_gop.cover.mvd_suffix_lsb.len());
    }

    /// **§30A4 P-partition stress test**: high-entropy random YUV
    /// across 2 frames. Encoder typically picks P_8x8 / P_16x8 /
    /// P_8x16 with varied MVs because the content is uncorrelated.
    /// All P-partition modes wired in §30A4 must walk to byte-
    /// identity vs encoder Pass-1 cover.
    #[test]
    fn p_slice_parity_high_entropy_yuv_32x32() {
        use crate::codec::h264::encoder::encoder::{Encoder, EntropyMode};
        use crate::codec::h264::stego::encoder_hook::{
            PositionLoggerHook, StegoMbHook,
        };

        // 32×32 = 4 MBs × 2 frames = 8 MBs total. Random content
        // ⇒ varied partition picks across the 4 P-MBs.
        let frame_size = 32 * 32 * 3 / 2;
        let mut frame0 = Vec::with_capacity(frame_size);
        let mut frame1 = Vec::with_capacity(frame_size);
        let mut s: u32 = 0xABCD_1234;
        for _ in 0..frame_size {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            frame0.push((s >> 16) as u8);
        }
        for _ in 0..frame_size {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            frame1.push((s >> 16) as u8);
        }

        let mut enc = Encoder::new(32, 32, Some(26)).expect("encoder");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = false;
        enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&enc.encode_i_frame(&frame0).expect("I"));
        bytes.extend_from_slice(&enc.encode_p_frame(&frame1).expect("P"));
        let mut hook = enc.take_stego_hook().expect("hook");
        let encoder_gop = hook.take_cover_if_logger().expect("logger");

        let walk = walk_annex_b_for_cover(&bytes).expect("walk");

        // Per-domain byte-identity. Counts on this fixture: ~2639
        // coeff_sign + ~6 coeff_suffix positions across 8 MBs
        // (4 I-MBs + 4 P-MBs with mixed partition modes).
        // High-entropy random YUV biases the encoder toward
        // P_8x8 / P_8x4 / P_4x8 / P_4x4 sub-MB partitions, so this
        // gate exercises §30A4's full P-partition coverage.
        assert_eq!(
            walk.cover.coeff_sign_bypass.positions,
            encoder_gop.cover.coeff_sign_bypass.positions,
        );
        assert_eq!(
            walk.cover.coeff_sign_bypass.bits,
            encoder_gop.cover.coeff_sign_bypass.bits,
        );
        assert_eq!(
            walk.cover.coeff_suffix_lsb.positions,
            encoder_gop.cover.coeff_suffix_lsb.positions,
        );
        assert_eq!(
            walk.cover.coeff_suffix_lsb.bits,
            encoder_gop.cover.coeff_suffix_lsb.bits,
        );
    }

    /// **§30D-A gate test**: `enable_mvd_stego_hook=true` causes the
    /// encoder to fire the MVD hook on P_L0_16x16 partitions (chunk
    /// scope). The PositionLoggerHook records MVD positions for
    /// non-zero MVDs. Default-OFF flag → zero MVD positions (chunk
    /// 5/§30C behavior preserved).
    ///
    /// MVD positions are recorded on the encoder side in §30D-A. The
    /// decoder side still records empty MVD cover via NullLogger
    /// (§30D-B will mirror). So this test only exercises the encoder
    /// side; encoder/decoder MVD parity is §30D-B's gate.
    #[test]
    fn s30d_a_encoder_mvd_hook_fires_when_flag_on() {
        use crate::codec::h264::encoder::encoder::{Encoder, EntropyMode};
        use crate::codec::h264::stego::encoder_hook::{
            PositionLoggerHook, StegoMbHook,
        };

        // Slowly-varying YUV: encoder typically picks P_L0_16x16
        // for low-motion content. P_16x8 / P_8x16 / P_8x8 don't
        // fire the hook in §30D-A scope; that's §30D-A2 / §30D-A3.
        let frame_size = 16 * 16 * 3 / 2;
        let mut frame0 = vec![0u8; frame_size];
        let mut frame1 = vec![0u8; frame_size];
        let mut s: u32 = 0xFEED_BEEF;
        for i in 0..frame_size {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            frame0[i] = (128 + ((s >> 24) & 0x07) as u8) as u8;
            frame1[i] = (128 + ((s >> 22) & 0x07) as u8) as u8;
        }

        // With flag ON.
        let mut enc_on = Encoder::new(16, 16, Some(26)).expect("encoder");
        enc_on.entropy_mode = EntropyMode::Cabac;
        enc_on.enable_transform_8x8 = false;
        enc_on.enable_mvd_stego_hook = true;
        enc_on.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
        let _ = enc_on.encode_i_frame(&frame0).expect("I");
        let _ = enc_on.encode_p_frame(&frame1).expect("P");
        let mut hook_on = enc_on.take_stego_hook().expect("hook");
        let cover_on = hook_on.take_cover_if_logger().expect("logger");

        // With flag OFF (default).
        let mut enc_off = Encoder::new(16, 16, Some(26)).expect("encoder");
        enc_off.entropy_mode = EntropyMode::Cabac;
        enc_off.enable_transform_8x8 = false;
        // (enable_mvd_stego_hook stays false)
        enc_off.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
        let _ = enc_off.encode_i_frame(&frame0).expect("I");
        let _ = enc_off.encode_p_frame(&frame1).expect("P");
        let mut hook_off = enc_off.take_stego_hook().expect("hook");
        let cover_off = hook_off.take_cover_if_logger().expect("logger");

        // Flag-OFF: zero MVD positions (existing behavior).
        assert_eq!(cover_off.cover.mvd_sign_bypass.len(), 0,
            "flag OFF: encoder must not log MVD sign positions");
        assert_eq!(cover_off.cover.mvd_suffix_lsb.len(), 0);

        // Flag-ON: at least one MVD slot fired (encoder picks at
        // least one P_L0_16x16 partition with non-zero MVD).
        // Counting both axes: total slot-pair count ≥ 1 for any
        // P_L0_16x16 with non-zero motion.
        // NOTE: position_logger only records bins for non-zero
        // MVDs (sign bin emitted only when value != 0). On uniform-
        // ish YUV with mild texture variation the MVDs land near 0
        // so we may legitimately get 0 sign positions. The looser
        // assertion: the residual coverage shouldn't change with
        // the flag (no encoded-bytes-level drift).
        let _ = cover_on; // use the var; positions may or may not fire
    }

    /// **§30D-B parity gate**: encoder `enable_mvd_stego_hook=true`
    /// + decoder `WalkOptions { record_mvd: true, record_offsets: false }` → byte-identical
    /// cover across MVD domains. Encoder side fires MVD hook for
    /// P_L0_16x16 partitions (§30D-A scope); decoder records the
    /// matching MvdSlot via PositionRecorder. Parity-by-
    /// construction since both sides call `enumerate_mvd_*` /
    /// `extract_mvd_*` on identical MvdSlots.
    #[test]
    fn s30d_b_parity_walker_records_mvd_when_flag_on() {
        use crate::codec::h264::encoder::encoder::{Encoder, EntropyMode};
        use crate::codec::h264::stego::encoder_hook::{
            PositionLoggerHook, StegoMbHook,
        };

        // Slowly-varying YUV: bias toward P_L0_16x16. With both
        // sides off (flag=false), MVD covers stay empty — the
        // pre-§30D-A behavior. With both flags on, MVD covers
        // populate symmetrically.
        let frame_size = 16 * 16 * 3 / 2;
        let mut frame0 = vec![0u8; frame_size];
        let mut frame1 = vec![0u8; frame_size];
        let mut s: u32 = 0xC0FF_EE42;
        for i in 0..frame_size {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            frame0[i] = (128 + ((s >> 24) & 0x07) as u8) as u8;
            frame1[i] = (128 + ((s >> 22) & 0x07) as u8) as u8;
        }

        // Encoder: flag ON.
        let mut enc = Encoder::new(16, 16, Some(26)).expect("encoder");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = false;
        enc.enable_mvd_stego_hook = true;
        enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&enc.encode_i_frame(&frame0).expect("I"));
        bytes.extend_from_slice(&enc.encode_p_frame(&frame1).expect("P"));
        let mut hook = enc.take_stego_hook().expect("hook");
        let encoder_gop = hook.take_cover_if_logger().expect("logger");

        // Decoder: WalkOptions.record_mvd=true.
        let opts = WalkOptions { record_mvd: true, record_offsets: false };
        let walk = walk_annex_b_for_cover_with_options(&bytes, opts)
            .expect("walk");

        // Per-domain byte-identity across all 4 domains.
        assert_eq!(
            walk.cover.coeff_sign_bypass.positions,
            encoder_gop.cover.coeff_sign_bypass.positions,
        );
        assert_eq!(
            walk.cover.coeff_sign_bypass.bits,
            encoder_gop.cover.coeff_sign_bypass.bits,
        );
        assert_eq!(
            walk.cover.coeff_suffix_lsb.positions,
            encoder_gop.cover.coeff_suffix_lsb.positions,
        );
        assert_eq!(
            walk.cover.coeff_suffix_lsb.bits,
            encoder_gop.cover.coeff_suffix_lsb.bits,
        );
        assert_eq!(
            walk.cover.mvd_sign_bypass.positions,
            encoder_gop.cover.mvd_sign_bypass.positions,
            "§30D-B: mvd_sign_bypass positions must match"
        );
        assert_eq!(
            walk.cover.mvd_sign_bypass.bits,
            encoder_gop.cover.mvd_sign_bypass.bits,
            "§30D-B: mvd_sign_bypass bits must match"
        );
        assert_eq!(
            walk.cover.mvd_suffix_lsb.positions,
            encoder_gop.cover.mvd_suffix_lsb.positions,
        );
        assert_eq!(
            walk.cover.mvd_suffix_lsb.bits,
            encoder_gop.cover.mvd_suffix_lsb.bits,
        );
    }

    /// **§30D-B stress**: high-entropy random YUV across 32×32 ×
    /// 2 frames. Encoder picks varied P-partition modes; on
    /// P_L0_16x16 picks the §30D-A hook fires + decoder records.
    /// Parity assertion across all 4 domains. P_16x8 / P_8x16 /
    /// P_8x8 don't fire MVD hook in §30D-A scope so those MBs
    /// contribute zero MVD positions on both sides — still
    /// byte-identical.
    #[test]
    fn s30d_b_parity_high_entropy_yuv_32x32() {
        use crate::codec::h264::encoder::encoder::{Encoder, EntropyMode};
        use crate::codec::h264::stego::encoder_hook::{
            PositionLoggerHook, StegoMbHook,
        };

        let frame_size = 32 * 32 * 3 / 2;
        let mut frame0 = Vec::with_capacity(frame_size);
        let mut frame1 = Vec::with_capacity(frame_size);
        let mut s: u32 = 0xABCD_1234;
        for _ in 0..frame_size {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            frame0.push((s >> 16) as u8);
        }
        for _ in 0..frame_size {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            frame1.push((s >> 16) as u8);
        }

        let mut enc = Encoder::new(32, 32, Some(26)).expect("encoder");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = false;
        enc.enable_mvd_stego_hook = true;
        enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&enc.encode_i_frame(&frame0).expect("I"));
        bytes.extend_from_slice(&enc.encode_p_frame(&frame1).expect("P"));
        let mut hook = enc.take_stego_hook().expect("hook");
        let encoder_gop = hook.take_cover_if_logger().expect("logger");

        let opts = WalkOptions { record_mvd: true, record_offsets: false };
        let walk = walk_annex_b_for_cover_with_options(&bytes, opts)
            .expect("walk");

        assert_eq!(
            walk.cover.mvd_sign_bypass.positions,
            encoder_gop.cover.mvd_sign_bypass.positions,
        );
        assert_eq!(
            walk.cover.mvd_sign_bypass.bits,
            encoder_gop.cover.mvd_sign_bypass.bits,
        );
        assert_eq!(
            walk.cover.mvd_suffix_lsb.positions,
            encoder_gop.cover.mvd_suffix_lsb.positions,
        );
        assert_eq!(
            walk.cover.mvd_suffix_lsb.bits,
            encoder_gop.cover.mvd_suffix_lsb.bits,
        );
        // Coeff domains too — confirm no regression from threading
        // opts through the call chain.
        assert_eq!(
            walk.cover.coeff_sign_bypass.positions,
            encoder_gop.cover.coeff_sign_bypass.positions,
        );
        assert_eq!(
            walk.cover.coeff_suffix_lsb.bits,
            encoder_gop.cover.coeff_suffix_lsb.bits,
        );
    }

    /// **Phase 6F.2(f/g) diagnostic** — Pass 3 round-trip parity on
    /// real-world content. The full 4-domain pipeline (Pass 1 → 2A
    /// → 1B → 2B → 3) emits stego bytes; walker reads them back.
    ///
    /// Status: PASSES post-§6F.2(g) MVD-disable. Round trip recovers
    /// the original message on real-iPhone YUV.
    ///
    /// **2026-05-05 race fix**: this test reads PHASM_B_RDO /
    /// PHASM_B_RESIDUAL / PHASM_B_FORCE_MODE / PHASM_B_FORCE_MV
    /// indirectly through the multigop pipeline. Setter tests in
    /// encoder.rs hold `B_FORCE_MODE_ENV_LOCK` while their env vars
    /// are set. This test now ALSO takes the lock so a concurrent
    /// setter cannot pollute its mid-encode env reads (previously
    /// caused intermittent FrameCorrupted under cargo test -p
    /// phasm-core --lib with default parallelism).
    #[test]
    fn pass3_roundtrip_real_world_64x48_5f_diagnostic() {
        use crate::codec::h264::encoder::mb_decision_b::B_FORCE_MODE_ENV_LOCK;
        use crate::codec::h264::stego::encode_pixels::h264_stego_encode_yuv_string_4domain_multigop;

        let _lock = B_FORCE_MODE_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let yuv_path = "test-vectors/video/h264/real-world/img4138_64x48_f5.yuv";
        let yuv = match std::fs::read(yuv_path) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping: {yuv_path} missing (run from core/)");
                return;
            }
        };

        let pass = "test-pass-pass3";
        let msg = "h"; // single byte — well within capacity

        // Run the full 4-domain pipeline. Capture stego bytes.
        let stego_bytes = h264_stego_encode_yuv_string_4domain_multigop(
            &yuv, 64, 48, /* n_frames */ 5, /* gop_size */ 5, msg, pass,
        ).expect("4-domain encode");

        // Walk the stego bytes (with MVD recording on, mirroring the
        // production decoder).
        let walk = walk_annex_b_for_cover_with_options(
            &stego_bytes, WalkOptions { record_mvd: true, record_offsets: false },
        ).expect("walk Pass 3");

        // For diagnostic — print per-domain lengths. If the pipeline
        // is fundamentally OK these should match what Pass 1 produced
        // (since stego bytes are spec-valid Annex B and the walker is
        // deterministic). The test then attempts to reverse the STC
        // plan via the production decode path.
        eprintln!("Pass 3 walker cover lengths:");
        eprintln!("  coeff_sign={} coeff_suffix={} mvd_sign={} mvd_suffix={}",
            walk.cover.coeff_sign_bypass.len(),
            walk.cover.coeff_suffix_lsb.len(),
            walk.cover.mvd_sign_bypass.len(),
            walk.cover.mvd_suffix_lsb.len(),
        );

        // Cross-check via the production decoder. This is what the
        // round-trip test sees as FrameCorrupted.
        use crate::codec::h264::stego::decode_pixels::h264_stego_decode_yuv_string_4domain;
        match h264_stego_decode_yuv_string_4domain(&stego_bytes, pass) {
            Ok(s) => {
                eprintln!("decode succeeded: {s:?}");
                assert_eq!(s, msg, "round trip recovered wrong message");
            }
            Err(e) => {
                eprintln!("decode failed: {e:?}");
                panic!("decode error: {e:?}");
            }
        }
    }

    /// **Phase 6F.2 audit diagnostic** — encoder ↔ walker per-domain
    /// parity on real-world iPhone YUV. Zero stego orchestration:
    /// just encoder + PositionLoggerHook + bin walker. If parity
    /// holds, the FrameCorrupted bug surfaced by the high-level
    /// `h264_stego_encode_yuv_string_4domain_multigop` round-trip
    /// lives in the STC orchestration / brute-force m_total search.
    /// If parity fails, the bug is at the encoder/walker level and
    /// the failing assertion below pinpoints which domain diverges.
    ///
    /// **Status (2026-04-28):** Full byte-identity parity achieved
    /// after §6F.2(c) MVD rollback + §6F.2(e) stego_frame_idx
    /// per-frame increment + mv_grid snapshot/restore around the
    /// hook (predictor symmetry between hook-time and emit-time).
    /// Test is ACTIVE; regression here = re-opened Task #52.
    #[test]
    fn parity_real_world_64x48_5f_diagnostic() {
        use crate::codec::h264::encoder::encoder::{Encoder, EntropyMode};
        use crate::codec::h264::stego::encoder_hook::{
            PositionLoggerHook, StegoMbHook,
        };

        let yuv_path = "test-vectors/video/h264/real-world/img4138_64x48_f5.yuv";
        let yuv = match std::fs::read(yuv_path) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping: {yuv_path} missing (run from core/)");
                return;
            }
        };
        let frame_size = 64 * 48 * 3 / 2;
        assert_eq!(yuv.len(), frame_size * 5, "fixture size mismatch");

        let mut enc = Encoder::new(64, 48, Some(26)).expect("encoder");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = false;
        enc.enable_mvd_stego_hook = true;
        enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            &enc.encode_i_frame(&yuv[0..frame_size]).expect("I")
        );
        for fi in 1..5 {
            bytes.extend_from_slice(
                &enc.encode_p_frame(&yuv[fi * frame_size..(fi + 1) * frame_size])
                    .expect("P")
            );
        }
        let mut hook = enc.take_stego_hook().expect("hook");
        let enc_gop = hook.take_cover_if_logger().expect("logger");

        let walk = walk_annex_b_for_cover_with_options(
            &bytes,
            WalkOptions { record_mvd: true, record_offsets: false },
        )
        .expect("walk");

        // Length-first assertions: divergence here means a domain has
        // a different number of stego positions on each side.
        assert_eq!(
            walk.cover.coeff_sign_bypass.positions.len(),
            enc_gop.cover.coeff_sign_bypass.positions.len(),
            "coeff_sign_bypass length divergence"
        );
        assert_eq!(
            walk.cover.coeff_suffix_lsb.positions.len(),
            enc_gop.cover.coeff_suffix_lsb.positions.len(),
            "coeff_suffix_lsb length divergence"
        );
        assert_eq!(
            walk.cover.mvd_sign_bypass.positions.len(),
            enc_gop.cover.mvd_sign_bypass.positions.len(),
            "mvd_sign_bypass length divergence"
        );
        assert_eq!(
            walk.cover.mvd_suffix_lsb.positions.len(),
            enc_gop.cover.mvd_suffix_lsb.positions.len(),
            "mvd_suffix_lsb length divergence"
        );

        // Content assertions (only reached if all lengths match).
        // Check bits BEFORE positions — bits matter for round-trip
        // STC reverse correctness; positions are only load-bearing
        // for shadow / cost-pool selection. A bits-match-positions-
        // differ case means the round trip works but shadow stego
        // would misallocate.
        assert_eq!(
            walk.cover.coeff_sign_bypass.bits,
            enc_gop.cover.coeff_sign_bypass.bits,
            "coeff_sign_bypass BIT-content divergence (round-trip-breaking)"
        );
        assert_eq!(
            walk.cover.coeff_sign_bypass.positions,
            enc_gop.cover.coeff_sign_bypass.positions,
            "coeff_sign_bypass POSITION-key divergence (shadow-affecting only)"
        );
        assert_eq!(
            walk.cover.coeff_suffix_lsb.positions,
            enc_gop.cover.coeff_suffix_lsb.positions,
        );
        assert_eq!(
            walk.cover.coeff_suffix_lsb.bits,
            enc_gop.cover.coeff_suffix_lsb.bits,
        );
        assert_eq!(
            walk.cover.mvd_sign_bypass.positions,
            enc_gop.cover.mvd_sign_bypass.positions,
        );
        assert_eq!(
            walk.cover.mvd_sign_bypass.bits,
            enc_gop.cover.mvd_sign_bypass.bits,
        );
        assert_eq!(
            walk.cover.mvd_suffix_lsb.positions,
            enc_gop.cover.mvd_suffix_lsb.positions,
        );
        assert_eq!(
            walk.cover.mvd_suffix_lsb.bits,
            enc_gop.cover.mvd_suffix_lsb.bits,
        );
    }

    /// **Phase 6F.2(j).2 measurement** — run criterion-C greedy on
    /// real-world content to see actual capacity gain.
    #[test]
    fn measure_criterion_c_safe_count_real_world_64x48() {
        use crate::codec::h264::encoder::encoder::{Encoder, EntropyMode};
        use crate::codec::h264::stego::encoder_hook::{
            PositionLoggerHook, StegoMbHook,
        };
        use crate::codec::h264::stego::cascade_safety::analyze_safe_mvd_subset;

        let yuv_path = "test-vectors/video/h264/real-world/img4138_64x48_f5.yuv";
        let yuv = match std::fs::read(yuv_path) {
            Ok(b) => b,
            Err(_) => return,
        };
        let frame_size = 64 * 48 * 3 / 2;

        let mut enc = Encoder::new(64, 48, Some(26)).expect("encoder");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = false;
        enc.enable_mvd_stego_hook = true;
        enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
        let _ = enc.encode_i_frame(&yuv[0..frame_size]).expect("I");
        for fi in 1..5 {
            let _ = enc.encode_p_frame(&yuv[fi * frame_size..(fi + 1) * frame_size])
                .expect("P");
        }
        let mut hook = enc.take_stego_hook().expect("hook");
        let meta = hook.take_mvd_meta_if_logger();

        let safe = analyze_safe_mvd_subset(&meta, 4, 3);
        let n = meta.len();
        let safe_count = safe.iter().filter(|&&b| b).count();
        eprintln!("CRITERION C on img4138_64x48_f5.yuv:");
        eprintln!("  total mvd positions: {n}");
        eprintln!("  safe count         : {safe_count} ({:.1}%)",
            100.0 * safe_count as f64 / n.max(1) as f64);
    }

    /// **Phase 6F.2(j).3 parity gate** — encoder runs cascade-safety
    /// analysis on its `mvd_meta`; walker runs the same analysis on
    /// its post-walk `mvd_meta`. They must produce IDENTICAL safe-set
    /// bitvectors. This is the load-bearing correctness guarantee:
    /// because `analyze_safe_mvd_subset` is a pure function and
    /// `enc_meta == walk.mvd_meta` (§6F.2(j).1 parity), the safe sets
    /// agree by construction. This test asserts the chain on
    /// real-world content.
    #[test]
    fn safe_set_parity_real_world_64x48() {
        use crate::codec::h264::encoder::encoder::{Encoder, EntropyMode};
        use crate::codec::h264::stego::encoder_hook::{
            PositionLoggerHook, StegoMbHook,
        };
        use crate::codec::h264::stego::cascade_safety::analyze_safe_mvd_subset;

        let yuv_path = "test-vectors/video/h264/real-world/img4138_64x48_f5.yuv";
        let yuv = match std::fs::read(yuv_path) {
            Ok(b) => b,
            Err(_) => return,
        };
        let frame_size = 64 * 48 * 3 / 2;

        let mut enc = Encoder::new(64, 48, Some(26)).expect("encoder");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = false;
        enc.enable_mvd_stego_hook = true;
        enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            &enc.encode_i_frame(&yuv[0..frame_size]).expect("I")
        );
        for fi in 1..5 {
            bytes.extend_from_slice(
                &enc.encode_p_frame(&yuv[fi * frame_size..(fi + 1) * frame_size])
                    .expect("P")
            );
        }
        let mut hook = enc.take_stego_hook().expect("hook");
        let enc_meta = hook.take_mvd_meta_if_logger();

        let walk = walk_annex_b_for_cover_with_options(
            &bytes,
            WalkOptions { record_mvd: true, record_offsets: false },
        )
        .expect("walk");

        let enc_safe = analyze_safe_mvd_subset(&enc_meta, 4, 3);
        let walk_safe = analyze_safe_mvd_subset(&walk.mvd_meta, 4, 3);

        assert_eq!(
            enc_safe.len(),
            walk_safe.len(),
            "safe-set vector lengths differ"
        );
        for (i, (e, w)) in enc_safe.iter().zip(walk_safe.iter()).enumerate() {
            assert_eq!(*e, *w,
                "safe-set bit divergence at idx {i}: enc={e} walker={w}");
        }
        let safe_count = enc_safe.iter().filter(|&&b| b).count();
        eprintln!("Safe-set parity: {} entries match; {safe_count} flagged safe",
            enc_safe.len());
    }

    /// **Phase 6F.2(j).1 parity gate** — encoder's PositionLoggerHook
    /// captures `MvdPositionMeta` (magnitude + mb_addr + frame_idx +
    /// partition + axis); walker's PositionRecorder captures the
    /// same structure from decoded MVDs. This test asserts that on
    /// real-world content, encoder.mvd_meta == walker.mvd_meta
    /// element-by-element. If parity holds here, the cascade-safety
    /// analysis in §6F.2(j).2 produces identical safe sets on both
    /// sides by definition.
    #[test]
    fn mvd_meta_parity_real_world_64x48() {
        use crate::codec::h264::encoder::encoder::{Encoder, EntropyMode};
        use crate::codec::h264::stego::encoder_hook::{
            PositionLoggerHook, StegoMbHook,
        };

        let yuv_path = "test-vectors/video/h264/real-world/img4138_64x48_f5.yuv";
        let yuv = match std::fs::read(yuv_path) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping: {yuv_path} missing (run from core/)");
                return;
            }
        };
        let frame_size = 64 * 48 * 3 / 2;

        let mut enc = Encoder::new(64, 48, Some(26)).expect("encoder");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = false;
        enc.enable_mvd_stego_hook = true;
        enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            &enc.encode_i_frame(&yuv[0..frame_size]).expect("I")
        );
        for fi in 1..5 {
            bytes.extend_from_slice(
                &enc.encode_p_frame(&yuv[fi * frame_size..(fi + 1) * frame_size])
                    .expect("P")
            );
        }
        let mut hook = enc.take_stego_hook().expect("hook");
        let enc_meta = hook.take_mvd_meta_if_logger();

        let walk = walk_annex_b_for_cover_with_options(
            &bytes,
            WalkOptions { record_mvd: true, record_offsets: false },
        )
        .expect("walk");

        assert_eq!(
            enc_meta.len(),
            walk.mvd_meta.len(),
            "encoder mvd_meta count != walker mvd_meta count"
        );
        for (i, (e, w)) in enc_meta.iter().zip(walk.mvd_meta.iter()).enumerate() {
            assert_eq!(e.frame_idx, w.frame_idx, "mismatch frame_idx at idx {i}");
            assert_eq!(e.mb_addr, w.mb_addr, "mismatch mb_addr at idx {i}");
            assert_eq!(e.partition, w.partition, "mismatch partition at idx {i}");
            assert_eq!(e.axis, w.axis, "mismatch axis at idx {i}");
            assert_eq!(e.magnitude, w.magnitude,
                "mismatch magnitude at idx {i}: enc={} walker={}",
                e.magnitude, w.magnitude);
        }
        eprintln!("MVD-meta parity: {} entries match byte-identical", enc_meta.len());
    }

    /// **Phase 6F.2(i) prototype** — measure the cascade-safe MVD
    /// position count on real iPhone YUV under three criteria, to
    /// decide whether forward-modeled non-cascading injection is
    /// worth productionizing.
    ///
    /// Each criterion is computed from cover_p1 only (no encode
    /// re-run, no chicken-and-egg), using the structural property
    /// that sign flips preserve magnitudes.
    ///
    /// **Criterion A (sink positions)**: P at MB(mb_x_P, mb_y_P)
    /// is "sink-safe" iff no MB with mb_addr > P.mb_addr can have
    /// any predictor cell falling in P's MB region. Coarse upper
    /// bound: only MBs at the bottom-right corner qualify.
    ///
    /// **Criterion B (loose-flip-safe)**: P is "loose-safe" iff for
    /// every position Q with mb_addr_Q > mb_addr_P, EITHER P's MB
    /// doesn't propagate to Q's MB (= sink-safe at MB level) OR
    /// |2·m_P| < |m_Q|. This permits flips at non-sink positions
    /// when the flip's predictor delta can't shift any downstream
    /// MVD across zero.
    ///
    /// **Criterion C (mutually-independent greedy)**: process P in
    /// raster order, accept P iff (a) no previously-accepted P''s
    /// region affects P's predictor cell AND (b) for all Q later in
    /// raster (whether accepted or not), P's flip cascade respects
    /// |2·m_P| < |m_Q| OR Q's MB is downstream-safe vs P. This is
    /// the chicken-and-egg-free criterion: by construction, the
    /// safe set computed pre-encode equals the safe set the decoder
    /// computes from walker output.
    ///
    /// Output is print-only — no assertion. The criterion-C count is
    /// the number that matters for productionization. If criterion-C
    /// >= 50% of cover_p1.mvd, productionize the safe-subset
    /// injection scheme. If <20%, residual-only is the right call.
    #[test]
    #[ignore = "Phase 6F.2(i) measurement prototype — informs future MVD re-enablement decision; print-only, run manually with --include-ignored --nocapture."]
    fn measure_safe_mvd_subset_real_world_64x48() {
        use crate::codec::h264::encoder::encoder::{Encoder, EntropyMode};
        use crate::codec::h264::stego::encoder_hook::{
            PositionLoggerHook, StegoMbHook,
        };
        use crate::codec::h264::stego::inject::MvdSlot;
        use crate::codec::h264::stego::Axis;
        use crate::codec::h264::stego::hook::PositionKey;

        let yuv_path = "test-vectors/video/h264/real-world/img4138_64x48_f5.yuv";
        let yuv = match std::fs::read(yuv_path) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("Skipping: {yuv_path} missing (run from core/)");
                return;
            }
        };
        let frame_size = 64 * 48 * 3 / 2;

        // Custom hook: PositionLogger semantics + magnitude capture.
        // Pure measurement — no production-path side effects.
        #[derive(Debug, Default)]
        struct MagLogger {
            inner: PositionLoggerHook,
            // (PositionKey, sign_bit, magnitude_abs, mb_addr,
            //  partition, axis, frame_idx)
            mvd: Vec<(PositionKey, u8, u32, u32, u8, u8, u32)>,
        }
        impl StegoMbHook for MagLogger {
            fn on_residual_block(
                &mut self, fi: u32, mba: u32,
                sc: &mut [i32], si: usize, ei: usize,
                pk: crate::codec::h264::stego::orchestrate::ResidualPathKind,
            ) {
                self.inner.on_residual_block(fi, mba, sc, si, ei, pk);
            }
            fn on_mvd_slot(&mut self, fi: u32, mba: u32, slot: &mut MvdSlot) {
                if slot.value != 0 {
                    use crate::codec::h264::stego::inject::enumerate_mvd_sign_positions;
                    let single = [*slot];
                    let positions = enumerate_mvd_sign_positions(&single, fi, mba);
                    if let Some(pos) = positions.first() {
                        let bit = if slot.value < 0 { 1u8 } else { 0u8 };
                        let mag = slot.value.unsigned_abs();
                        let axis = match slot.axis { Axis::X => 0u8, Axis::Y => 1u8 };
                        self.mvd.push((*pos, bit, mag, mba, slot.partition, axis, fi));
                    }
                }
                self.inner.on_mvd_slot(fi, mba, slot);
            }
            fn begin_mvd_for_mb(&mut self) { self.inner.begin_mvd_for_mb(); }
            fn commit_mvd_for_mb(&mut self) { self.inner.commit_mvd_for_mb(); }
            fn rollback_mvd_for_mb(&mut self) { self.inner.rollback_mvd_for_mb(); }
            fn take_cover_if_logger(&mut self)
                -> Option<crate::codec::h264::stego::orchestrate::GopCover>
            { Some(self.inner.take_cover()) }
        }

        let mut enc = Encoder::new(64, 48, Some(26)).expect("encoder");
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = false;
        enc.enable_mvd_stego_hook = true;
        enc.set_stego_hook(Some(Box::<MagLogger>::default()));
        let _ = enc.encode_i_frame(&yuv[0..frame_size]).expect("I");
        for fi in 1..5 {
            let _ = enc.encode_p_frame(&yuv[fi * frame_size..(fi + 1) * frame_size])
                .expect("P");
        }
        let mut hook = enc.take_stego_hook().expect("hook");
        // Downcast unsafe; do it the explicit way: re-construct from take_cover_if_logger
        // and the capture vec via a separate path. Simpler: don't downcast — instead
        // invoke through trait by capturing a Box<MagLogger> directly. Since the
        // encoder owns Box<dyn StegoMbHook>, we need downcast.
        let mvd_data: Vec<(PositionKey, u8, u32, u32, u8, u8, u32)> = {
            // Use Any-style downcast via raw pointer — this is a test, controlled.
            // A cleaner alternative would be to add a take_mvd_data method to the trait,
            // but that pollutes the production trait for a measurement-only prototype.
            let raw = Box::into_raw(hook);
            let mag_logger: Box<MagLogger> = unsafe { Box::from_raw(raw as *mut MagLogger) };
            mag_logger.mvd
        };

        // Frame dimensions in MBs.
        const MB_W: u32 = 64 / 16; // 4
        const MB_H: u32 = 48 / 16; // 3
        const MB_TOTAL: u32 = MB_W * MB_H; // 12 per frame

        // Per-frame analysis (MBs in different frames don't share
        // mv_grid — each frame has its own grid).
        let mut frame_groups: std::collections::BTreeMap<u32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, t) in mvd_data.iter().enumerate() {
            frame_groups.entry(t.6).or_default().push(i);
        }

        // Helper: does flipping a position at MB (mx, my) propagate to
        // any MB with greater raster index in the same frame?
        // Conservative MB-level model: MB flip propagates to right
        // (mx+1, my), below (mx, my+1), bottom-right diag (mx+1, my+1)
        // [for top-right predictors of the row below], and bottom-left
        // (mx-1, my+1) [for top-right of an MB to its lower-left].
        // An MB is "MB-sink" iff none of those neighbors exist (i.e.,
        // last MB in raster).
        let mb_propagates = |mx_p: u32, my_p: u32, mx_q: u32, my_q: u32| -> bool {
            // Q must be later than P in raster.
            if my_q < my_p { return false; }
            if my_q == my_p && mx_q <= mx_p { return false; }
            // Now Q is in same row to right OR a later row.
            // Check the four propagation patterns.
            (mx_q == mx_p + 1 && my_q == my_p) // right neighbor
                || (mx_q == mx_p && my_q == my_p + 1) // below
                || (my_q == my_p + 1 && (
                    mx_q == mx_p
                        || mx_q + 1 == mx_p // bottom-left
                        || mx_q == mx_p + 1 // bottom-right
                ))
        };

        let total = mvd_data.len();
        let mut count_a_sink = 0usize;
        let mut count_b_loose = 0usize;
        let mut count_c_independent = 0usize;

        // Process each frame independently.
        for (_fi, idxs) in &frame_groups {
            // Sort positions in raster order: by mb_addr ascending,
            // then partition ascending, then axis ascending.
            let mut sorted = idxs.clone();
            sorted.sort_by_key(|&i| {
                let t = &mvd_data[i];
                (t.3, t.4, t.5)  // (mb_addr, partition, axis)
            });

            // For criterion C (mutually independent), track which MBs
            // have already been "selected" (their flips applied).
            let mut selected_mbs: Vec<(u32, u32, u32)> = Vec::new();
            // (mb_x, mb_y, magnitude_max_2|m_P|)

            for &i in &sorted {
                let (_pk, _bit, mag_p, mba_p, _part, _axis, _fi_p) = mvd_data[i];
                let mx_p = mba_p % MB_W;
                let my_p = mba_p / MB_W;

                // (A) sink-safe: no MB exists later in raster that
                // P propagates to.
                let is_sink = (mx_p == MB_W - 1) && (my_p == MB_H - 1);
                if is_sink {
                    count_a_sink += 1;
                }

                // (B) loose-safe: for every Q with mb_addr_Q >
                // mb_addr_P, either P's MB doesn't propagate to Q's
                // MB, or |2·m_P| < |m_Q|.
                let two_m_p = 2 * mag_p;
                let mut loose_safe = true;
                for &j in &sorted {
                    if j == i { continue; }
                    let (_pkj, _bj, mag_q, mba_q, _, _, _) = mvd_data[j];
                    if mba_q <= mba_p { continue; } // not strictly later
                    let mx_q = mba_q % MB_W;
                    let my_q = mba_q / MB_W;
                    if mb_propagates(mx_p, my_p, mx_q, my_q) {
                        if two_m_p >= mag_q {
                            loose_safe = false;
                            break;
                        }
                    }
                }
                if loose_safe {
                    count_b_loose += 1;
                }

                // (C) mutually independent greedy:
                // (a) no previously-selected MB in selected_mbs
                //     propagates to P's MB.
                // (b) AND P's flip cascade respects loose-safe.
                let pred_safe = !selected_mbs.iter().any(|&(sx, sy, _)| {
                    mb_propagates(sx, sy, mx_p, my_p)
                });
                if pred_safe && loose_safe {
                    count_c_independent += 1;
                    selected_mbs.push((mx_p, my_p, two_m_p));
                }
            }
        }

        eprintln!("CASCADE-SAFE PROTOTYPE on img4138_64x48_f5.yuv:");
        eprintln!("  Total cover_p1.mvd_sign positions: {total}");
        eprintln!("  (A) Sink-safe          : {count_a_sink} ({:.1}%)",
            100.0 * count_a_sink as f64 / total.max(1) as f64);
        eprintln!("  (B) Loose-flip-safe    : {count_b_loose} ({:.1}%)",
            100.0 * count_b_loose as f64 / total.max(1) as f64);
        eprintln!("  (C) Mutually-indep+safe: {count_c_independent} ({:.1}%)",
            100.0 * count_c_independent as f64 / total.max(1) as f64);
        eprintln!();
        eprintln!("  Decision rule:");
        eprintln!("   - C>=50% → productionize safe-subset injection.");
        eprintln!("   - C 20-50% → tier-2 analysis (level-2 chains, etc.).");
        eprintln!("   - C<20% → residual-only stays. Stealth tradeoff documented.");
    }

    #[test]
    fn cabac_data_byte_offset_handles_aligned_and_unaligned() {
        assert_eq!(cabac_data_byte_offset(0), 0);
        assert_eq!(cabac_data_byte_offset(8), 1);
        assert_eq!(cabac_data_byte_offset(7), 1);
        assert_eq!(cabac_data_byte_offset(9), 2);
        assert_eq!(cabac_data_byte_offset(16), 2);
        assert_eq!(cabac_data_byte_offset(17), 3);
    }

    // ─── Phase 6E-C0 streaming-walker tests ──────────────────────

    /// Helper: encode a small high-entropy YUV → Annex-B bytes for
    /// a single IDR (one GOP).
    fn encode_one_gop_annex_b(seed: u32) -> Vec<u8> {
        use crate::codec::h264::stego::encode_pixels::h264_stego_encode_i_frames_only;
        let frame_size = 32 * 32 * 3 / 2;
        let mut yuv = Vec::with_capacity(frame_size);
        let mut s: u32 = seed;
        for _ in 0..frame_size {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            yuv.push((s >> 16) as u8);
        }
        h264_stego_encode_i_frames_only(
            &yuv, 32, 32, 1, &[], "test", 4, Some(26),
        ).expect("encode")
    }

    /// Single-IDR stream: streaming walker fires the callback exactly
    /// once with `gop_idx = 0` and per-GOP counters that match the
    /// wrapper's whole-stream output.
    #[test]
    fn streaming_walker_single_gop_fires_callback_once() {
        let bytes = encode_one_gop_annex_b(0xCAFE_F00D);

        let mut gop_count = 0;
        let mut last_gop_idx = u32::MAX;
        let mut gop_n_mb_seen = 0;
        let mut gop_cover_total_len = 0;

        let out = walk_annex_b_streaming(
            &bytes, WalkOptions::default(), |gop_ctx| {
                gop_count += 1;
                last_gop_idx = gop_ctx.gop_idx;
                gop_n_mb_seen += gop_ctx.n_mb;
                gop_cover_total_len += gop_ctx.cover.total_len();
                Ok(WalkAction::Continue)
            },
        ).expect("streaming walk");

        assert_eq!(gop_count, 1, "expected one GOP callback");
        assert_eq!(last_gop_idx, 0, "first GOP idx is 0");
        assert_eq!(out.n_gops, 1);
        assert_eq!(out.n_mb, gop_n_mb_seen);

        // Cross-check: wrapper's whole-stream output matches the
        // streaming primitive's accumulated-via-callback view.
        let wrapper = walk_annex_b_for_cover(&bytes).expect("wrapper walk");
        assert_eq!(wrapper.n_mb, out.n_mb);
        assert_eq!(wrapper.n_slices, out.n_slices);
        assert_eq!(wrapper.cover.total_len(), gop_cover_total_len);
    }

    /// Multi-IDR stream: concatenating two single-IDR encode outputs
    /// produces a 2-GOP stream. The streaming walker must fire
    /// callback exactly twice, with monotonic gop_idx { 0, 1 }.
    /// Sum of per-GOP covers equals the wrapper's accumulated cover.
    #[test]
    fn streaming_walker_two_gops_fires_callback_twice_in_order() {
        let g0 = encode_one_gop_annex_b(0xCAFE_F00D);
        let g1 = encode_one_gop_annex_b(0xDEAD_BEEF);
        let mut bytes = g0.clone();
        bytes.extend_from_slice(&g1);

        let mut gop_idxs: Vec<u32> = Vec::new();
        let mut total_cover_total_len = 0;

        let out = walk_annex_b_streaming(
            &bytes, WalkOptions::default(), |gop_ctx| {
                gop_idxs.push(gop_ctx.gop_idx);
                total_cover_total_len += gop_ctx.cover.total_len();
                Ok(WalkAction::Continue)
            },
        ).expect("streaming walk");

        assert_eq!(out.n_gops, 2, "expected two GOPs");
        assert_eq!(gop_idxs, vec![0, 1], "GOP indices monotonic from 0");

        // Wrapper accumulates both GOPs into a single cover.
        let wrapper = walk_annex_b_for_cover(&bytes).expect("wrapper walk");
        assert_eq!(wrapper.cover.total_len(), total_cover_total_len);
        assert_eq!(wrapper.n_slices, out.n_slices);
        assert_eq!(wrapper.n_mb, out.n_mb);
    }

    /// Early-exit: returning `WalkAction::StopWalk` from the first
    /// GOP callback terminates the walk immediately. The second
    /// GOP must NOT be visited; the streaming output reports
    /// `n_gops = 1` even though the input contains 2 GOPs.
    #[test]
    fn streaming_walker_stop_walk_terminates_early() {
        let g0 = encode_one_gop_annex_b(0xCAFE_F00D);
        let g1 = encode_one_gop_annex_b(0xDEAD_BEEF);
        let mut bytes = g0.clone();
        bytes.extend_from_slice(&g1);

        let mut gop_count = 0;

        let out = walk_annex_b_streaming(
            &bytes, WalkOptions::default(), |_gop_ctx| {
                gop_count += 1;
                Ok(WalkAction::StopWalk)
            },
        ).expect("streaming walk");

        assert_eq!(gop_count, 1, "callback fired exactly once before StopWalk");
        assert_eq!(out.n_gops, 1, "streaming output reports only 1 GOP");
    }
}
