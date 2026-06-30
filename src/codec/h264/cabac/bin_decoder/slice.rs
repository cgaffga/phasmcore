// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Top-level decode-side slice walker.
//
// Public entry points: `walk_annex_b_for_cover(annex_b)` (and the
// pre-parsed / streaming variants `walk_nalus_streaming_with_options`
// etc.) → `CoverWalkOutput { cover: DomainCover, n_mb, n_slices, .. }`.
// Walks the bitstream end-to-end and emits a per-domain `DomainCover`
// (CoeffSign / CoeffSuffixLsb / MvdSign / MvdSuffixLsb) for STC
// extract. The cover bits are derived through the same shared
// `enumerate_*_positions` / `extract_*_bits` /
// `record_residual_block_into_cover` primitives (in
// `crate::codec::h264::stego`) that the encode-side cover capture
// uses, so the two sides agree on positions + bit order
// by-construction on identical (scan_coeffs, MvdSlot) inputs.
//
// **Scope**: the complete walker. Parses SPS / PPS / slice headers
// (latest-wins, single-active assumption), initializes the CABAC
// engine + contexts per slice, consumes cabac_alignment_one_bit to
// the byte boundary, and dispatches every MB type across I/SI, P/SP,
// and B slices (I_16x16, I_NxN/I_4x4/I_8x8, P_Skip + P-partitions +
// intra-in-P, and the full B mb_type tree incl. partitioned/B_8x8/
// intra-in-B), decoding residuals and recording per-domain positions.
// I_PCM is the only MB type still unwired. CABAC-only — CAVLC PPS is
// rejected (`WalkError::NotCabac`). Optional MVD recording
// (`WalkOptions::record_mvd`) is opt-in; default off.

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
use crate::codec::h264::tables::BLOCK_INDEX_TO_POS;
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
    /// Slice header indicated an unsupported type (none of I/SI,
    /// P/SP, B).
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
    /// Per-MVD-position metadata aligned by index with
    /// `cover.mvd_sign_bypass.positions`. Empty when
    /// `WalkOptions { record_mvd: false }`.
    pub mvd_meta: Vec<crate::codec::h264::stego::hook::MvdPositionMeta>,
    /// Frame dimensions in macroblocks, parsed from the first SPS
    /// encountered. Used by
    /// `cascade_safety::analyze_safe_mvd_subset` at decode time.
    /// 0/0 when no slice was walked (degenerate input).
    pub mb_w: u32,
    pub mb_h: u32,
}

/// Walker configuration knobs. Default-everything-off keeps the
/// walk behavior unchanged (MVD recording off).
#[derive(Debug, Clone, Copy, Default)]
pub struct WalkOptions {
    /// When true, the slice walker records MVD positions + bits
    /// into the per-domain cover (mvd_sign_bypass
    /// + mvd_suffix_lsb). Must match how the stego embed was built:
    /// the cover used at extract has to include the MVD domains iff
    /// the embed did, otherwise the encode and decode covers diverge.
    pub record_mvd: bool,
}

/// Streaming-walker callback verdict. Returned by the per-GOP
/// `on_gop` callback to either continue walking the next
/// GOP or terminate the walk early. Early-exit is the load-bearing
/// optimization for shadow decode: once a parity-tier candidate
/// successfully RS-decodes + AES-GCM-SIV-decrypts a shadow's
/// payload, no further GOPs need to be walked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkAction {
    Continue,
    StopWalk,
}

/// Per-GOP context delivered to the streaming walker's callback.
/// Owns the GOP's `DomainCover` (positions + bits) and
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
    /// Per-MVD-position metadata aligned by index with
    /// `cover.mvd_sign_bypass.positions`. Empty when
    /// `WalkOptions { record_mvd: false }`.
    pub mvd_meta: Vec<crate::codec::h264::stego::hook::MvdPositionMeta>,
    /// Frame dimensions in macroblocks (from the active SPS at the
    /// time this GOP was walked). Used by
    /// `cascade_safety::analyze_safe_mvd_subset`.
    pub mb_w: u32,
    pub mb_h: u32,
}

/// Streaming-walker summary. Returned at end-of-stream
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
/// **Scope**: walks I/SI, P/SP, and B slices; rejects CAVLC
/// (CABAC-only); parses SPS+PPS in stream order with latest-wins
/// selection.
pub fn walk_annex_b_for_cover(annex_b: &[u8]) -> Result<CoverWalkOutput, WalkError> {
    walk_annex_b_for_cover_with_options(annex_b, WalkOptions::default())
}

/// Variant of `walk_annex_b_for_cover` that accepts explicit
/// `WalkOptions`. Used by consumers that opt into MVD recording.
pub fn walk_annex_b_for_cover_with_options(
    annex_b: &[u8],
    opts: WalkOptions,
) -> Result<CoverWalkOutput, WalkError> {
    // PHASM_PROFILE — the pure-Rust CABAC cover walk (per-GOP; runs in every
    // encode pass + the round-trip verify + decode).
    let _prof = crate::codec::h264::profile::scope("cover_walk");
    let trace = std::env::var("PHASM_PERF_TRACE")
        .map(|v| v == "1")
        .unwrap_or(false);
    let t_nal = if trace { Some(std::time::Instant::now()) } else { None };
    let nalus = parse_nal_units_annexb(annex_b)?;
    let dt_nal = t_nal.map(|t| t.elapsed());

    let t_walk = if trace { Some(std::time::Instant::now()) } else { None };
    let result = walk_nalus_for_cover_with_options(&nalus, opts);
    let dt_walk = t_walk.map(|t| t.elapsed());

    if trace {
        if let (Some(d_nal), Some(d_walk)) = (dt_nal, dt_walk) {
            let total_us = (d_nal.as_micros() + d_walk.as_micros()).max(1) as f64;
            eprintln!(
                "[PHASM_PERF_TRACE walker] annex_b={} bytes nalus={}",
                annex_b.len(), nalus.len(),
            );
            eprintln!(
                "[PHASM_PERF_TRACE]   {:<22} {:>9.1} ms  ({:>5.1}%)",
                "parse_nal_units_annexb",
                d_nal.as_secs_f64() * 1000.0,
                d_nal.as_micros() as f64 / total_us * 100.0,
            );
            eprintln!(
                "[PHASM_PERF_TRACE]   {:<22} {:>9.1} ms  ({:>5.1}%)",
                "walk_nalus + slice MBs",
                d_walk.as_secs_f64() * 1000.0,
                d_walk.as_micros() as f64 / total_us * 100.0,
            );
        }
    }
    result
}

/// Variant of `walk_nalus_for_cover` that accepts explicit
/// `WalkOptions`. Thin wrapper over `walk_nalus_streaming_with_options`
/// that accumulates per-GOP covers into a single whole-stream cover
/// — preserves all current parity gates as regression tests.
pub fn walk_nalus_for_cover_with_options(
    nalus: &[NalUnit],
    opts: WalkOptions,
) -> Result<CoverWalkOutput, WalkError> {
    // The parallel walker handles all cover-walk cases.
    walk_nalus_for_cover_parallel(nalus, opts)
}


/// Parallel cover walk via per-slice rayon `par_iter`.
///
/// **Two-pass design:**
/// 1. **Linear pre-pass**: scan nalus, parse SPS/PPS, build a
///    `Vec<SliceCtx>` with each slice's frame_idx, nal_idx, parsed
///    header, active SPS/PPS clones. Cheap (~30 slices × ~50 µs
///    each at 1080p × 30f).
/// 2. **Parallel walk**: each slice walks its own MBs into a local
///    `PositionRecorder` (with `reserve_for_mb_count` pre-sized).
///    CABAC arithmetic engine resets per slice (slice header starts
///    fresh state) + neighbours are slice-local → no cross-slice
///    dependency.
/// 3. **Sequential merge**: concat per-slice covers in original slice
///    order. Preserves the byte-identical position+bit emit order of
///    the sequential walker.
///
/// Memory: per-thread local recorder ≈ 200 KB × N_threads. Well inside
/// the per-GOP streaming memory bound.
///
/// WASM target falls back to sequential (`rayon::par_iter` requires
/// threading; WASM threading is opt-in via `Atomics` proposal which
/// most consumers don't enable). Same for builds without the
/// `parallel` Cargo feature.
pub fn walk_nalus_for_cover_parallel(
    nalus: &[NalUnit],
    opts: WalkOptions,
) -> Result<CoverWalkOutput, WalkError> {

    // Pre-pass: gather per-slice ctxs.
    let prepared = prepare_slice_ctxs(nalus)?;
    if prepared.slices.is_empty() {
        return Err(WalkError::NoSlices);
    }

    // Parallel slice walk. Each slice writes into a local recorder.
    let slice_results: Vec<SliceWalkResult> = {
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            use rayon::prelude::*;
            prepared
                .slices
                .par_iter()
                .map(|ctx| walk_one_slice_into_local(ctx, &opts))
                .collect::<Result<Vec<_>, _>>()?
        }
        #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
        {
            prepared
                .slices
                .iter()
                .map(|ctx| walk_one_slice_into_local(ctx, &opts))
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    // Sequential merge in slice order. The `extend_from` calls walk
    // each per-slice cover Vec once; merge cost ≈ same as the
    // sequential walker's incremental push (we just move the
    // ordering boundary).
    let n_slices = slice_results.len();
    let mut acc_cover = DomainCover::default();
    let mut acc_mvd_meta: Vec<crate::codec::h264::stego::hook::MvdPositionMeta>
        = Vec::new();
    let mut total_n_mb = 0usize;
    // Pre-size the merged Vecs to avoid log2 reallocations during the
    // concat phase.
    let total_cs_bypass: usize = slice_results
        .iter()
        .map(|r| r.cover.coeff_sign_bypass.len())
        .sum();
    let total_cs_lsb: usize = slice_results
        .iter()
        .map(|r| r.cover.coeff_suffix_lsb.len())
        .sum();
    let total_mvd_sign: usize = slice_results
        .iter()
        .map(|r| r.cover.mvd_sign_bypass.len())
        .sum();
    let total_mvd_lsb: usize = slice_results
        .iter()
        .map(|r| r.cover.mvd_suffix_lsb.len())
        .sum();
    let total_mvd_meta: usize = slice_results.iter().map(|r| r.mvd_meta.len()).sum();
    acc_cover.coeff_sign_bypass.reserve(total_cs_bypass);
    acc_cover.coeff_suffix_lsb.reserve(total_cs_lsb);
    acc_cover.mvd_sign_bypass.reserve(total_mvd_sign);
    acc_cover.mvd_suffix_lsb.reserve(total_mvd_lsb);
    acc_mvd_meta.reserve(total_mvd_meta);

    for r in slice_results {
        acc_cover.extend_from(r.cover);
        acc_mvd_meta.extend(r.mvd_meta);
        total_n_mb += r.n_mb;
    }

    Ok(CoverWalkOutput {
        cover: acc_cover,
        n_mb: total_n_mb,
        n_slices,
        mvd_meta: acc_mvd_meta,
        mb_w: prepared.mb_w as u32,
        mb_h: prepared.mb_h as u32,
    })
}

/// Per-slice context built by the linear pre-pass. Each entry owns
/// the parsed header + cloned SPS/PPS so parallel slice walks need
/// no shared state.
struct SliceCtx {
    nal_idx: u32,
    frame_idx: u32,
    rbsp: Vec<u8>,
    header: SliceHeader,
    pps: Pps,
    mb_w: usize,
    mb_count: usize,
}

struct PreparedSlices {
    slices: Vec<SliceCtx>,
    mb_w: usize,
    mb_h: usize,
}

struct SliceWalkResult {
    cover: DomainCover,
    mvd_meta: Vec<crate::codec::h264::stego::hook::MvdPositionMeta>,
    n_mb: usize,
}

/// Linear pre-pass: parse SPS/PPS as they arrive, build a per-slice
/// context list. Frame indices are assigned in VCL-NAL order
/// (matches the streaming walker's `frame_idx.wrapping_add(1)`
/// semantics), nal indices match the original `nalus[]` position.
fn prepare_slice_ctxs(nalus: &[NalUnit]) -> Result<PreparedSlices, WalkError> {
    let mut active_sps: Option<Sps> = None;
    let mut active_pps: Option<Pps> = None;
    let mut slices = Vec::new();
    let mut frame_idx: u32 = 0;
    let mut last_mb_w = 0usize;
    let mut last_mb_h = 0usize;

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
                let mb_w = sps.pic_width_in_mbs as usize;
                let mb_h = sps.pic_height_in_map_units as usize
                    * if sps.frame_mbs_only_flag { 1 } else { 2 };
                let mb_count = mb_w * mb_h;
                last_mb_w = mb_w;
                last_mb_h = mb_h;

                slices.push(SliceCtx {
                    nal_idx,
                    frame_idx,
                    rbsp: nal.rbsp.clone(),
                    header,
                    pps: pps.clone(),
                    mb_w,
                    mb_count,
                });
                frame_idx = frame_idx.wrapping_add(1);
            }
            _ => { /* AUD, SEI, filler — skip */ }
        }
    }

    Ok(PreparedSlices {
        slices,
        mb_w: last_mb_w,
        mb_h: last_mb_h,
    })
}

/// Per-slice CABAC walk — mirrors the body of the VCL branch in
/// `walk_nalus_streaming_with_options`. Writes into a thread-local
/// `PositionRecorder` that the caller takes ownership of.
fn walk_one_slice_into_local(
    ctx: &SliceCtx,
    opts: &WalkOptions,
) -> Result<SliceWalkResult, WalkError> {
    let mut recorder = PositionRecorder::new();
    // Pre-size cover Vecs from per-slice mb_count.
    recorder.reserve_for_mb_count(ctx.mb_count);

    let cabac_byte_off = cabac_data_byte_offset(ctx.header.data_bit_offset);
    if cabac_byte_off > ctx.rbsp.len() {
        return Err(WalkError::H264(H264Error::UnexpectedEof));
    }
    let cabac_bytes = &ctx.rbsp[cabac_byte_off..];

    let slot = pick_init_slot(&ctx.header);
    let mut dec =
        CabacDecoder::new_slice(cabac_bytes, slot, ctx.header.slice_qp, ctx.mb_w)?;

    let n_mb = walk_slice_mbs(
        &mut dec,
        &ctx.header,
        &ctx.pps,
        ctx.mb_count,
        ctx.frame_idx,
        ctx.mb_w,
        ctx.nal_idx,
        &mut recorder,
        opts,
    )?;

    Ok(SliceWalkResult {
        cover: recorder.take_cover(),
        mvd_meta: recorder.take_mvd_meta(),
        n_mb,
    })
}

/// Per-GOP streaming walker (Annex-B input). Parses the byte
/// stream, emits one `GopContext` per GOP via the
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

/// Per-GOP streaming walker (pre-parsed NAL list input). Same
/// semantics as `walk_annex_b_streaming` but lets the
/// caller share a NAL parser run with other consumers.
pub fn walk_nalus_streaming_with_options<F>(
    nalus: &[NalUnit],
    opts: WalkOptions,
    mut on_gop: F,
) -> Result<StreamingWalkOutput, WalkError>
where
    F: FnMut(GopContext) -> Result<WalkAction, WalkError>,
{
    let trace = std::env::var("PHASM_PERF_TRACE")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut t_setup_total = std::time::Duration::ZERO;
    let mut t_walk_mbs_total = std::time::Duration::ZERO;
    let mut t_gop_extract_total = std::time::Duration::ZERO;

    let mut active_sps: Option<Sps> = None;
    let mut active_pps: Option<Pps> = None;
    let mut recorder = PositionRecorder::new();
    // Pre-size the cover Vecs once after the first SPS arrives,
    // eliminating the ~log2(N) doubling reallocs on hot
    // fixtures (1080p × 30f IPPPP has ~514k CoeffSign positions).
    // `reserved_for` tracks the largest mb_count we've sized for so
    // multi-GOP streams don't down-size or re-reserve needlessly.
    let mut reserved_for: usize = 0;
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
                // B-slices accepted; the walker dispatches into
                // `walk_b_mb`, which decodes the full B mb_type tree
                // (B_Skip, B_Direct_16x16, B_L0/L1/Bi_16x16,
                // partitioned 4..=21, B_8x8, and intra-in-B 23..=47).
                // I_PCM (48) is the only B mb_type still unwired.

                // GOP boundary detection: an IDR after at least one
                // VCL NAL in the current GOP closes the prior GOP.
                if t.is_idr() && had_vcl_in_current_gop {
                    let t_gop = if trace { Some(std::time::Instant::now()) } else { None };
                    let cover = recorder.take_cover();
                    let mvd_meta = recorder.take_mvd_meta();
                    if let Some(t) = t_gop { t_gop_extract_total += t.elapsed(); }
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
                    // next VCL NAL (the `= true` at the end of this VCL
                    // branch); no explicit reset needed.
                }

                let t_setup = if trace { Some(std::time::Instant::now()) } else { None };
                let mb_w = sps.pic_width_in_mbs as usize;
                let mb_h = sps.pic_height_in_map_units as usize
                    * if sps.frame_mbs_only_flag { 1 } else { 2 };
                let mb_count = mb_w * mb_h;

                // First-slice cover Vec pre-sizing. The recorder
                // starts with zero-cap Vecs; without this, the
                // per-MB CABAC loop pushes into doubling Vecs and pays
                // ~24 MB cumulative memcopy across log2(514k) ≈ 19
                // reallocs at 1080p × 30f. Idempotent: re-arms only on
                // larger mb_count (multi-resolution streams).
                if mb_count > reserved_for {
                    recorder.reserve_for_mb_count(mb_count);
                    reserved_for = mb_count;
                }

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
                if let Some(t) = t_setup { t_setup_total += t.elapsed(); }

                let t_mbs = if trace { Some(std::time::Instant::now()) } else { None };
                let n_mb = walk_slice_mbs(
                    &mut dec, &header, pps, mb_count, frame_idx, mb_w,
                    nal_idx, &mut recorder, &opts,
                )?;
                if let Some(t) = t_mbs { t_walk_mbs_total += t.elapsed(); }

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
        let t_gop = if trace { Some(std::time::Instant::now()) } else { None };
        let cover = recorder.take_cover();
        let mvd_meta = recorder.take_mvd_meta();
        if let Some(t) = t_gop { t_gop_extract_total += t.elapsed(); }
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
        })?;
        total_n_gops += 1;
    }

    if total_n_slices == 0 {
        return Err(WalkError::NoSlices);
    }

    if trace {
        let total_us = (t_setup_total.as_micros()
            + t_walk_mbs_total.as_micros()
            + t_gop_extract_total.as_micros())
            .max(1) as f64;
        eprintln!(
            "[PHASM_PERF_TRACE walker_inner] n_mb={} n_slices={} n_gops={}",
            total_n_mb, total_n_slices, total_n_gops,
        );
        let report = |label: &str, d: std::time::Duration| {
            let ms = d.as_secs_f64() * 1000.0;
            let pct = d.as_micros() as f64 / total_us * 100.0;
            eprintln!("[PHASM_PERF_TRACE]   {label:<22} {ms:>9.1} ms  ({pct:>5.1}%)");
        };
        report("slice header + cabac init", t_setup_total);
        report("walk_slice_mbs (per-MB)", t_walk_mbs_total);
        report("GOP record extract", t_gop_extract_total);
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
/// Exposed pub so the bitstream-mod splicer can convert a captured
/// engine-local bit offset to NAL-RBSP-absolute coordinates:
/// `rbsp_byte = cabac_data_byte_offset + engine_bit / 8`.
pub fn cabac_data_byte_offset(data_bit_offset: usize) -> usize {
    data_bit_offset.div_ceil(8)
}

/// Per-MB walking loop. Dispatches per slice type:
/// - I/SI: I_16x16, I_NxN (I_4x4 / I_8x8); I_PCM errors.
/// - P/SP: P_SKIP, P-partitions (mb_type 0..3), and intra-in-P;
///   I_PCM-in-P errors.
/// - B: full B mb_type tree via `walk_b_mb` (see there).
///
/// Inverts the spec macroblock_layer() / mb_pred() / sub_mb_pred()
/// syntax (§ 7.3.5) that the OpenH264 emitter produces, in CABAC
/// parsing order.
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

        // Call `neighbors.new_row()` at every MB-row boundary so
        // left-neighbour availability resets per spec § 6.4.9.
        if let Some(prev_y) = prev_mb_y
            && mb_y != prev_y {
                dec.neighbors.new_row();
            }
        prev_mb_y = Some(mb_y);

        // num_ref_idx_l0_active drives whether walker expects
        // ref_idx_l0 unary bins per partition. At
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

/// P-slice MB dispatch. Routes P_SKIP, intra-in-P, and P_partition
/// (mb_type 0..3, dispatched to `walk_p_partition_mb`).
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
    // 1. mb_skip_flag.
    let is_skip = decode_mb_skip_flag(dec, mb_x)?;
    if is_skip {
        // P_SKIP: no further syntax — read end_of_slice_flag and
        // commit PSkip neighbors. No residuals, no MVDs → zero stego
        // coverage.
        let is_last = decode_end_of_slice_flag(dec)?;
        let nb = CabacNeighborMB {
            mb_type: MbTypeClass::PSkip,
            mb_skip_flag: true,
            ..CabacNeighborMB::default()
        };
        dec.neighbors.commit(mb_x, nb);
        return Ok(is_last);
    }

    // 2. mb_type_p.
    let mb_type_p = decode_mb_type_p(dec, mb_x)?;
    if mb_type_p <= 3 {
        // P-partition: P_L0_16x16 / P_L0_16x8 / P_L0_8x16 / P_8x8.
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
    // Decompose mb_type per spec § 7.3.5 / Table 7-11.
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
    let dc_scan = decode_residual_block_recording(
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
            let ac_scan = decode_residual_block_recording(
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
            let dc_flat = decode_residual_block_recording(
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
                let ac_scan = decode_residual_block_recording(
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

/// B-slice MB dispatch. Decodes the full B mb_type tree:
/// B_Skip (mb_skip_flag = 1), B_Direct_16x16 (0), B_L0/L1/Bi_16x16
/// (1/2/3), partitioned (4..=21), B_8x8 (22), and intra-in-B
/// (23..=47). I_PCM (mb_type = 48) is the only still-unwired case.
///
/// B_Skip semantics (spec § 7.4.5):
/// - mb_skip_flag = 1 → no further syntax for this MB.
/// - The MB's L0/L1 motion is reconstructed by spatial direct (or
///   temporal direct, per `direct_spatial_mv_pred_flag`) from
///   neighbors — no MVD/ref_idx/residual bins are on the wire.
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

    // 2. mb_type via the B-slice bin tree (covers all 24
    //    spec values 0..=23). Per-mb_type body dispatch below.
    let mb_type = decode_mb_type_b(dec, mb_x)?;
    let mb_addr_u32 = (mb_y * mb_w + mb_x) as u32;

    // Mirror the encoder's transform_size_8x8_flag emission gate.
    // For all B-MB inter
    // modes phasm ships (16x16/16x8/8x16/B_8x8 with sub_mb_type ≤ 3),
    // noSubMbPartSizeLessThan8x8Flag = 1 (no sub-8x8 partitions).
    // Encoder emits the flag iff `pps.transform_8x8_mode_flag &&
    // cbp_luma != 0`; walker must consume it accordingly.
    let t8x8 = pps.transform_8x8_mode_flag;

    // Per-mb_type dispatcher. The L0/L1/Bi/partitioned/B_8x8
    // walkers decode non-zero CBP residuals via the shared
    // `finish_b_inter` tail.
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
/// `B_Direct_16x16` (mb_type = 0) walker. Reads CBP and, when
/// non-zero, decodes residuals via the [`decode_residual_block_cabac`]
/// helpers (same code path as P). Direct has no MVDs on the wire.
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
    // B_Direct_16x16 CAN have CBP > 0 per spec § 7.4.5.1. An earlier
    // walker hardcoded CBP=0 + WalkError::Unsupported for any
    // non-zero CBP, which prevented Direct+residual emission on the
    // encoder side.
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
    // Direct/Skip has no L0 ref_idx on the wire (derived from
    // spatial-direct neighbour read). Commit 0 for the entire MB;
    // spec § 7.4.5.1 says inferred ref_idxs are not part of
    // neighbour state lookup.
    finish_b_inter_with_mb_type(
        dec, mb_x, prev_mb_qp,
        /* l0_abs */ None, /* l1_abs */ None,
        frame_idx, nal_idx, mb_addr_u32, recorder, t8x8,
        MbTypeClass::BSkipOrDirect,
        [0i8; 16],
    )
}

/// `B_L0_16x16` (mb_type = 1) walker. One L0 MVD pair.
/// `ref_idx_l0` decoded when num_active_l0 > 1. Non-zero CBP
/// decodes residuals (luma 4×4 + chroma DC/AC) via the shared
/// `finish_b_inter` tail.
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
    // Capture the decoded ref_idx_l0 so the neighbour commit uses
    // the actual on-wire value (not 0).
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

/// `B_L1_16x16` (mb_type = 2). Mirror of L0 path, using L1
/// neighbour state for bin0 ctxIdxInc and committing
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
    // L1_16x16 has no L0 ref_idx on the wire (partition is L1-only
    // per Table 7-14). Commit [0;16].
    finish_b_inter(
        dec, mb_x, prev_mb_qp,
        None, Some([[abs_x; 16], [abs_y; 16]]),
        frame_idx, nal_idx, mb_addr_u32, recorder, t8x8,
        [0i8; 16],
    )
}

/// `B_Bi_16x16` (mb_type = 3). Two MVD pairs (L0 then L1 per spec
/// § 7.3.5.1 mb_pred order). Each list uses its own neighbour
/// state for bin0 ctxIdxInc. `ref_idx_l0` decoded when
/// num_active_l0 > 1 (Bi uses L0 list).
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
    // Capture decoded ref_idx_l0 (Bi uses L0).
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

/// Partitioned B mb_type walker (mb_types 4..21).
/// Looks up `(shape, list_usage_part0, list_usage_part1)` via
/// `b_partitioned::partitioned_b_meta`, then decodes per spec
/// § 7.3.5.1 mb_pred order (partition 0's MVDs L0-then-L1 before
/// partition 1's). Non-zero CBP residuals are decoded via
/// `finish_b_inter`'s shared luma 4×4 + chroma DC/AC path (spec
/// § 7.3.5.3 residual() at the partitioned MB).
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
    use crate::codec::h264::b_partition_meta::{
        partitioned_b_meta, BListUse,
    };

    let meta = partitioned_b_meta(mb_type as u32).ok_or_else(|| {
        WalkError::H264(H264Error::Unsupported(format!(
            "walk_b_partitioned: mb_type {mb_type} not in 4..=21"
        )))
    })?;

    // ref_idx_l0 per partition per spec § 7.3.5.1 mb_pred(): all
    // ref_idx_l0 BEFORE MVDs, in partition-index order, filtered by
    // uses-L0 (skip if partition is L1-only). Capture decoded
    // values per partition for the neighbour commit fill below.
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

    // Within-MB MVD tracker per-list, per spec § 9.3.3.1.1.7.
    // Partition 1's bin 0 ctxIdxInc reads partition 0's
    // just-decoded MVD via
    // `compute_mvd_ctx_idx_inc_bin0_per_list`.
    //
    // H.264 spec § 7.3.5.1 / § 9.3.3.1.1 interprets B-slice
    // partitioned MVDs in **list-major** order: outer iterates
    // list 0..2, inner iterates partition index 0..N-1, decoding
    // an MVD pair only when that partition uses that list. The walker
    // reads them in that order to stay in sync with the emitter.
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

    // Pass the full per-sub-MB MVD layout (not a broadcast of the
    // max magnitude) so the per-position neighbour fill matches what
    // the emitter produced. Otherwise the
    // next MB's bin0 ctxIdxInc reads from a position the emitter left
    // at 0 but the walker filled with max — CABAC desync at the bin
    // level.
    let l0_for_finish = if any_uses_l0(meta) { Some(current_mvd.comp) } else { None };
    let l1_for_finish = if any_uses_l1(meta) { Some(current_mvd.comp_l1) } else { None };
    // Per-block ref_idx_l0 fill from partition geometry (spec
    // § 6.4.2.2 partition layout) so the neighbour ctxIdxInc for the
    // next MB matches a spec decoder.
    let ref_idx_l0_array = walker_fill_ref_idx_l0_partitioned(meta, ref_idx_l0_decoded);
    finish_b_inter(
        dec, mb_x, prev_mb_qp, l0_for_finish, l1_for_finish,
        frame_idx, nal_idx, mb_addr_u32, recorder, t8x8,
        ref_idx_l0_array,
    )
}

/// `B_8x8` walker (mb_type = 22). Decodes:
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
/// 3. coded_block_pattern. Non-zero CBP is decoded via
///    `finish_b_inter` (luma 4×4 + chroma DC/AC), per spec
///    § 7.3.5.3 residual().
/// 4. end_of_slice_flag.
///
/// Aggregate MVD magnitudes across sub-MBs (max per axis per list)
/// for the neighbour commit, per spec § 9.3.3.1.1.7.
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

    // ref_idx_l0 per sub-MB per spec § 7.3.5.1 mb_pred(): all
    // ref_idx_l0 AFTER 4×sub_mb_type and BEFORE MVDs, in
    // sub-MB-index order, filtered by sub-MB-uses-L0
    // (Direct=0 + L1=2 skip; L0=1 + Bi=3 emit). Capture decoded
    // values per sub-MB + within-MB ref_idx tracker for spec
    // § 6.4.11.7 ctxIdxInc lookups.
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

    // Within-MB MVD tracker, per spec § 9.3.3.1.1.7.
    //
    // The spec parses B_8x8 sub-MB MVDs in **list-major** order
    // (H.264 spec § 7.3.5.1 + § 9.3.3.1.1): outer iterates list
    // 0..2, inner iterates sub-MB index 0..4, decoding an MVD
    // pair only when that sub-MB uses that list. The walker reads
    // them in that order to stay in sync with the emitter.
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

    // Pass per-sub-MB MVD layout (mirror of walk_b_partitioned
    // above).
    let any_l0 = sub_mb_types.iter().any(|&s| matches!(s, 1 | 3));
    let any_l1 = sub_mb_types.iter().any(|&s| matches!(s, 2 | 3));
    let l0_some = if any_l0 { Some(current_mvd.comp) } else { None };
    let l1_some = if any_l1 { Some(current_mvd.comp_l1) } else { None };
    // Per-block ref_idx_l0 fill from sub-MB geometry (spec
    // § 6.4.2.2 sub-MB partition layout).
    let ref_idx_l0_array = walker_fill_ref_idx_l0_b8x8(sub_mb_types, ref_idx_l0_decoded);
    finish_b_inter(
        dec, mb_x, prev_mb_qp, l0_some, l1_some,
        frame_idx, nal_idx, mb_addr_u32, recorder, t8x8,
        ref_idx_l0_array,
    )
}

/// True if any partition's list usage includes L0 (= L0 or Bi).
fn any_uses_l0(meta: crate::codec::h264::b_partition_meta::BPartitionedMeta) -> bool {
    use crate::codec::h264::b_partition_meta::BListUse;
    matches!(meta.part0, BListUse::L0 | BListUse::Bi)
        || matches!(meta.part1, BListUse::L0 | BListUse::Bi)
}

/// True if any partition's list usage includes L1.
fn any_uses_l1(meta: crate::codec::h264::b_partition_meta::BPartitionedMeta) -> bool {
    use crate::codec::h264::b_partition_meta::BListUse;
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

/// Decode one X+Y MVD pair for a B-slice partition with
/// within-MB-aware bin 0 ctxIdxInc derivation.
/// `(cur_bx, cur_by)` are the partition's TL 4×4-block coordinates
/// within the MB. For partition 1 of 16×8/8×16 (and sub-MBs 1+ of
/// B_8x8), this consults `current_mvd` (per-list) before falling
/// back to cross-MB neighbours, per spec § 9.3.3.1.1.7.
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

/// Shared tail for B-inter walkers: read CBP, decode
/// `mb_qp_delta` + residual blocks (luma 4×4 + chroma DC + chroma
/// AC) when `cbp != 0`, commit neighbour as inter (with per-list
/// MVD magnitudes), read `end_of_slice_flag`.
///
/// The CBP=0 fast path is preserved; the non-zero CBP path mirrors
/// `walk_p_inter_residual`'s 4×4 luma + chroma DC/AC decode (no 8×8
/// transform support — the B-side encoder stays on 4×4 transform).
///
/// `l0_abs` / `l1_abs` are `Option<[[i16; 16]; 2]>` — full
/// per-position arrays so partitioned + B_8x8 walkers can
/// reproduce the emitter's per-sub-MB neighbour layout (instead of
/// the broadcast that 16x16 walkers use). When sub-MB MVDs differ
/// across the MB (e.g. B_8x8 with one non-zero sub-MB and three
/// zero), the broadcast was overstating magnitude at zero positions
/// → next-MB bin0 ctxIdxInc disagreed with the emitter → CABAC desync.
#[allow(clippy::too_many_arguments)]
/// Builds a 16-entry per-4×4-block ref_idx_l0 array from
/// per-partition decoded values, following the spec § 6.4.2.2
/// partition layout so neighbour ctxIdxInc matches a spec decoder.
fn walker_fill_ref_idx_l0_partitioned(
    meta: crate::codec::h264::b_partition_meta::BPartitionedMeta,
    ref_idx_l0: [u8; 2],
) -> [i8; 16] {
    use crate::codec::h264::b_partition_meta::BListUse;
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

/// Per-sub-MB ref_idx_l0 fill (spec § 6.4.2.2 sub-MB partition
/// layout).
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

/// Walker-side helper for P-slice partition neighbour ref_idx_l0
/// fill. P-partitions all use L0.
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

    // Mirror the encoder's transform_size_8x8_flag emission gate.
    // For all B-MB inter modes phasm ships
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
                let scan = decode_residual_block_recording(
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
            let dc_flat = decode_residual_block_recording(
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
                let ac_scan = decode_residual_block_recording(
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

    // Per spec § 9.3.3.1.1.7: commit per-list MVD magnitudes so
    // subsequent neighbour bin0 ctxIdxInc reads the right
    // list-specific state.
    let abs_mvd_l0 = l0_abs.unwrap_or([[0; 16]; 2]);
    let abs_mvd_l1 = l1_abs.unwrap_or([[0; 16]; 2]);
    let mut nb = CabacNeighborMB {
        mb_type: mb_type_class,
        mb_skip_flag: false,
        cbp_luma,
        cbp_chroma,
        // Propagate the per-block ref_idx_l0 captured from the wire
        // so next MB's ref_idx ctxIdxInc matches what a
        // spec-conforming decoder computes (the prior [0;16]
        // hardcode desynced vs the reference decoder whenever ref_idx>0).
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

/// I_NxN macroblock walker. Inverts the spec I_NxN
/// macroblock_layer() / residual() syntax (§ 7.3.5). Returns the
/// end_of_slice_flag bit.
///
/// **Scope**: handles both I_4x4 (transform_size_8x8_flag = 0; 16
/// luma blocks of cat=2) and I_8x8 (flag = 1; 4 luma blocks of
/// cat=5), gated on `pps.transform_8x8_mode_flag` per spec.
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

    // 2. Per-block intra prediction modes. I_4x4 has 16 mode flags
    //    (one per 4×4 block); I_8x8 has 4 (one per 8×8 block) — the
    //    contexts are shared (Table 9-39 specifies ctxIdxOffsets
    //    68/69 for both I_4x4 and I_8x8).
    let n_pred_modes = if use_8x8 { 4 } else { 16 };
    for _ in 0..n_pred_modes {
        let prev_flag = decode_prev_intra4x4_pred_mode_flag(dec)?;
        if !prev_flag {
            let _rem = decode_rem_intra4x4_pred_mode(dec)?;
        }
    }

    // 3. intra_chroma_pred_mode.
    let chroma_pred_mode = decode_intra_chroma_pred_mode(dec, mb_x)?;

    // 4. coded_block_pattern. Returns the chroma:luma packed byte.
    let cbp_byte = decode_coded_block_pattern(dec, mb_x)?;
    let cbp_luma = cbp_byte & 0x0F;
    let cbp_chroma = (cbp_byte >> 4) & 0x03;

    // 5. mb_qp_delta — ONLY when cbp_value != 0 (spec § 7.3.5).
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
        // 6a. I_8x8 luma residual: 4 blocks of cat=5 (no per-block
        //     CBF). Each block has 64 coefficients (start=0, end=63).
        //     Gated per 8×8 block via cbp_luma bit k. Set every 4×4
        //     block within an 8×8-coded block as coded in the cat=2
        //     CBF state so subsequent I_4x4 MBs see consistent
        //     neighbor CBF.
        for k in 0..4usize {
            if cbp_luma & (1 << k) != 0 {
                let path_kind = ResidualPathKind::Luma8x8 {
                    block_idx: k as u8,
                };
                // Scoped so the recorder borrow is free for the
                // `on_residual_block` call below. The inline logger is a
                // zero-cost NullLogger (positions are enumerated from the
                // decoded coefficients after the fact).
                let mut null = NullLogger;
                let scan = {
                    let logger: &mut dyn PositionLogger = &mut null;
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
            // Mark all four 4×4 sub-blocks coded/uncoded per the 8×8
            // bit. Necessary so the next MB's
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
                let scan = decode_residual_block_recording(
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
            let dc_flat = decode_residual_block_recording(
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
                let ac_scan = decode_residual_block_recording(
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

    // 10. Commit neighbor state for next MB. The
    //     transform_size_8x8_flag bit drives the next MB's
    //     transform_size_8x8_flag CABAC ctx (Table 9-39 ctxIdxInc
    //     derivation).
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

/// Unified P-partition walker. Dispatches MVD decode based on
/// `mb_type_p` (0..3); residual + neighbor commit shared via
/// `decode_p_residuals_and_finish`.
///
/// Inverts the spec P-slice mb_pred() / sub_mb_pred() MVD syntax
/// (§ 7.3.5.1 / § 7.3.5.2).
///
/// MVD positions are recorded into the mvd_sign_bypass /
/// mvd_suffix_lsb cover domains only when `WalkOptions::record_mvd`
/// is set (via `decode_one_mvd_pair_p` → `recorder.on_mvd_slot`);
/// otherwise a `NullLogger` drops the inline emissions (zero-cost).
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

    // 1. P_8x8 carries 4 sub_mb_types BEFORE any MVDs (spec
    //    § 7.3.5.2). Other partition types skip this step.
    let sub_types: Option<[u32; 4]> = if mb_type_p == 3 {
        let mut sm = [0u32; 4];
        for s in sm.iter_mut() {
            *s = decode_sub_mb_type_p(dec)?;
        }
        Some(sm)
    } else {
        None
    };

    // 2. ref_idx_l0 per partition per spec § 7.3.5.1 mb_pred().
    // At num_ref_idx_l0_active=1 (single-ref default) the gate is
    // closed and zero bins read. P-slice partitions all use L0 by
    // definition (B-slice list-usage filtering doesn't apply).
    // Capture decoded values per partition for the neighbour commit
    // fill below.
    let mut ref_idx_l0_p = [0u8; 4];
    // Within-MB tracker so partition 1+ reads partition 0's
    // just-decoded ref_idx for ctxIdxInc lookups (spec § 6.4.11.7).
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

    // 3. MVDs per partition (spec § 7.3.5.1 mb_pred order):
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
    // sub_mb_type == 0 (P_L0_8x8).
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

/// Decode the MVD pairs for a P-partition MB, per spec
/// § 7.3.5.1 mb_pred() / § 7.3.5.2 sub_mb_pred().
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
            // P_16x8: 2 partitions stacked. Spec partition_id =
            // mbPartIdx*4 + subMbPartIdx (subMb=0 for non-8x8 modes).
            // The partition_ids must be {0, 4} (NOT {0, 1}) to match
            // the spec-aligned partition_id used by the encoder hook key.
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, 0, 0, 4, 2, 0,
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, 0, 2, 4, 2, 4,
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
        }
        2 => {
            // P_8x16: 2 partitions side-by-side. Same spec rule as
            // above — partition_id = mbPartIdx*4 (subMb=0).
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, 0, 0, 2, 4, 0,
                frame_idx, nal_idx, mb_addr_u32, recorder, opts,
            )?;
            decode_one_mvd_pair_p(
                dec, current_mvd, mb_x, 2, 0, 2, 4, 4,
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

/// Decode the MVD pairs for one 8×8 sub-MB, per spec § 7.3.5.2
/// sub_mb_pred().
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
/// sub_part_idx`.
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
/// for subsequent same-MB partition neighbor lookups. Inverts the
/// spec UEG3 mvd_l0/mvd_l1 binarization (§ 9.3.2.3).
///
/// When `opts.record_mvd` is true, record the decoded MVD value
/// via `PositionRecorder::on_mvd_slot` for both x and y axes.
/// Cover parity with the encode side is by-construction: both
/// route MvdSlots through the shared `enumerate_mvd_*_positions` /
/// `extract_mvd_*_bits` primitives in `stego::inject` on identical
/// slots.
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
    let mvd_x = {
        let logger: &mut dyn PositionLogger = &mut null;
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
        let logger: &mut dyn PositionLogger = &mut null;
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

    // Record MVD slots for the X and Y axes when the walker is
    // opt'd in (consumed by the shared
    // `enumerate_mvd_*_positions` / `extract_mvd_*_bits` primitives
    // in `stego::inject`).
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

    // 5. transform_size_8x8_flag — present iff PPS+cbp_luma>0+
    //    no_sub_mb_part_size_lt_8x8 (spec § 7.3.5).
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
        // 8×8 luma residual: 4 cat=5 blocks (no per-block CBF), per
        // spec § 7.3.5.3 residual().
        for k in 0..4usize {
            if cbp_luma & (1 << k) != 0 {
                let path_kind = ResidualPathKind::Luma8x8 {
                    block_idx: k as u8,
                };
                // Scoped so the recorder borrow is free for the
                // `on_residual_block` call below. The inline logger is a
                // zero-cost NullLogger (positions are enumerated from the
                // decoded coefficients after the fact).
                let mut null = NullLogger;
                let scan = {
                    let logger: &mut dyn PositionLogger = &mut null;
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
            // Mark all four 4×4 sub-blocks coded/uncoded per the 8×8
            // bit so subsequent 4×4 MBs see the same neighbor CBF
            // state.
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
                let scan = decode_residual_block_recording(
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
            let dc_flat = decode_residual_block_recording(
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
                let ac_scan = decode_residual_block_recording(
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
    // Propagate decoded ref_idx_l0 per partition geometry so
    // spec-conforming decoders agree on the next MB's ref_idx
    // ctxIdxInc.
    nb.ref_idx_l0 = ref_idx_l0_array;
    nb.abs_mvd_comp = current_mvd.to_neighbor();
    nb.transform_size_8x8_flag = use_8x8;
    dec.neighbors.commit(mb_x, nb);

    Ok(is_last)
}

/// Helper: decode one residual block and record its bypass-bin
/// positions into the recorder's cover. The inline decoder logger is a
/// `NullLogger` (zero-cost): the recorder's enumerate-from-coeffs path
/// gives identical positions to an inline logger by construction
/// (encoder-side parity gate, chunk 6A), so position recording is done
/// from the decoded coefficients after the fact rather than inline.
#[allow(clippy::too_many_arguments)]
fn decode_residual_block_recording(
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
    let logger: &mut dyn PositionLogger = &mut null;
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

    #[test]
    fn cabac_data_byte_offset_handles_aligned_and_unaligned() {
        assert_eq!(cabac_data_byte_offset(0), 0);
        assert_eq!(cabac_data_byte_offset(8), 1);
        assert_eq!(cabac_data_byte_offset(7), 1);
        assert_eq!(cabac_data_byte_offset(9), 2);
        assert_eq!(cabac_data_byte_offset(16), 2);
        assert_eq!(cabac_data_byte_offset(17), 3);
    }
}
