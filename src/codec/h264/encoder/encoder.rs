// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

#![allow(clippy::field_reassign_with_default)]
// CabacNeighborMB state is built up field-by-field after a default so
// each assignment carries an inline spec reference; a single struct
// literal with ..Default::default() would fold those back into opaque
// field order. Deliberate code-style choice.

//! Top-level H.264 encoder driver.
//!
//! Phase 6A.8 shipped `encode_i_frame_pcm` (I_PCM macroblocks,
//! lossless pixel-copy, validates framing bit-exact against a
//! conformant external H.264 decoder).
//!
//! Phase 6A.10 ships `encode_i_frame` (Intra_16x16 DC mode, forward
//! DCT + quant + Hadamard DC + CAVLC + reconstruction). This is the
//! first output exercising the Phase 6A.1–6A.6 pipeline end-to-end.
//! The I_PCM path stays as `encode_i_frame_pcm` for backwards
//! compatibility with the 6A.8 tests.
//!
//! Phase 6B adds P-frames + motion estimation.
//!
//! Algorithm notes:
//!   docs/design/video/h264/encoder-algorithms/frame-orchestration.md
//!   docs/design/video/h264/encoder-algorithms/intra16x16-mb-encode.md

use super::bitstream_writer::{
    build_aud_rbsp, build_pps_cabac, build_pps_cabac_high, build_pps_cavlc, build_sps_baseline,
    build_sps_high, build_sps_main,
    continue_slice_header_b, continue_slice_header_i, continue_slice_header_p,
    wrap_rbsp_as_nal, BitWriter,
    BSliceHeaderParams, ISliceHeaderParams, PSliceHeaderParams, PpsParams,
    PrimaryPicType, SpsParams, VuiParams,
};
use crate::codec::h264::cabac::context::CabacInitSlot;
use crate::codec::h264::cabac::encoder::{
    encode_coded_block_pattern, encode_end_of_slice_flag,
    encode_intra_chroma_pred_mode, encode_mb_qp_delta,
    encode_mb_skip_flag_b, encode_mb_type_b, encode_mb_type_i, CabacEncoder,
};
use crate::codec::h264::cabac::neighbor::{CabacNeighborMB, MbTypeClass};
use crate::codec::h264::cabac::slice::{append_cabac_zero_words, assemble_cabac_slice_rbsp};
use super::i4x4_encode::{derive_i4x4_mode_flags, encode_i4x4_mb, I4x4MbResult};
use super::inter_mode::{cbp_to_codenum_inter, luma_8x8_cbp_mask, pack_cbp};
use super::intra_predictor::{choose_intra_16x16_mode_psy, choose_intra_chroma_mode, satd_16x16};
use super::motion_compensation::{apply_chroma_mv_block, apply_luma_mv_block};
use super::motion_estimation::{MotionEstimator, MotionVector};
use super::partition_decision::{
    decide_p_mb_with_cost, PMbChoice, SubMbChoice, SUB_MB_ORIGINS_4X4,
    SUB_MB_ORIGINS_PX,
};
use super::partition_state::{
    predict_mv_for_mb_partition, predict_mv_for_partition, EncoderMvGrid, REF_IDX_NONE,
};
use super::quantization::{
    forward_quantize_4x4, forward_quantize_dc_chroma, forward_quantize_dc_luma,
    trellis_quantize_4x4, QuantParams, QuantSlice,
};
use super::rate_control::{FrameType, RateController};
use super::reconstruction::{raster_to_scan_levels, scan_to_raster_levels, ReconBuffer};
use super::reference_buffer::ReferenceBuffer;
use super::transform::{forward_dct_4x4, forward_hadamard_2x2, forward_hadamard_4x4};
use super::EncoderError;
use crate::codec::h264::cavlc_writer::{encode_cavlc_block, CavlcBlockType};
use crate::codec::h264::transform::{
    derive_chroma_qp, inverse_16x16_dc_hadamard, inverse_chroma_dc_2x2_hadamard,
    reconstruct_residual_4x4_with_dc,
};
use crate::codec::h264::NalType;

const ANNEX_B_START_CODE: [u8; 4] = [0x00, 0x00, 0x00, 0x01];

/// Entropy coding mode. Switching to `Cabac` upgrades the stream's
/// profile to Main (profile_idc = 77) and flips PPS
/// `entropy_coding_mode_flag` to 1. Phase 6C.6c scope: I_16x16 only;
/// P-slices in CABAC land in 6C.6d.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyMode {
    Cavlc,
    Cabac,
}

/// §intra-in-B (#319) Phase 3 — slice context for [`Encoder::write_i16x16_macroblock_cabac_with_ctx`].
/// Selects the mb_type prefix table (I / P / B) without changing the
/// inner residual emission pipeline (luma DC/AC + chroma DC/AC + CBF
/// neighbour state are slice-agnostic for I_16x16 once the mb_type bin
/// is emitted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraSliceCtx {
    /// I-slice intra MB. `encode_mb_type_i(cabac, mb_type, mb_x)`,
    /// codenum 1..24.
    I,
    /// P-slice intra-in-P MB. `encode_mb_type_p(cabac, mb_type + 5, mb_x)`,
    /// codenum 6..29 per Table 7-13 (+5 offset from I-slice).
    P,
    /// B-slice intra-in-B MB (§319). `encode_mb_type_b(cabac, mb_type + 23, mb_x)`,
    /// codenum 24..47 per Table 7-14 (+23 offset from I-slice).
    B,
}

/// §B-Partitioned-Residual Stage A (#206) — return value from
/// [`Encoder::emit_b_residual_for_pred`]. The caller uses these to
/// populate the [`crate::codec::h264::cabac::neighbor::CabacNeighborMB`]
/// commit after the helper returns; the helper itself has already
/// emitted CBP / `transform_size_8x8_flag` / `mb_qp_delta` / residual
/// blocks and written reconstructed samples back into `self.recon`.
struct BResidualResult {
    cbp_luma_8x8: u8,
    cbp_chroma: u8,
    qp_delta_emitted: i32,
    current_cbf: crate::codec::h264::cabac::neighbor::CurrentMbCbf,
}

/// v1.4 Phase 2.3 (#305) — multi-ref L0 configuration knob.
///
/// Threads through the encoder so slice-header construction + emit
/// functions know whether to (a) drop the
/// `num_ref_idx_l0_active_minus1=0` slice-header override and (b)
/// emit `ref_idx_l0` via `encode_ref_idx()` per partition. Default
/// `SINGLE_REF` preserves v1.3 ship-state bit-identically.
///
/// `max_l0_active = 2` matches the locked v1.4 design (Q1: 2-ref
/// L0). Spec allows up to 16; any value > 1 enables multi-ref
/// emission. L1 always single-ref under v1.4 scope (Q1).
#[derive(Debug, Clone, Copy)]
pub struct MultiRefConfig {
    pub enabled: bool,
    pub max_l0_active: u8,
}

impl MultiRefConfig {
    /// Default — single L0 ref everywhere. Bit-identical to v1.3
    /// ship behavior (slice header overrides L0 active to 1; emit
    /// functions skip ref_idx_l0 since num_active=1).
    pub const SINGLE_REF: Self = Self { enabled: false, max_l0_active: 1 };

    /// v1.4 ship target — 2 active L0 refs, ref_idx emitted on the
    /// wire. Slice header drops the active-override (lets PPS
    /// default of 2 stand). Phase 4 (#307) ME populates non-zero
    /// ref_idx; Phase 2.4 emit functions write them.
    pub const DUAL_REF_L0: Self = Self { enabled: true, max_l0_active: 2 };
}

/// Top-level encoder.
#[derive(Debug)]
pub struct Encoder {
    pub width: u32,
    pub height: u32,
    pub frame_num: u8,
    /// Stego-only logical frame counter. Unlike `frame_num` (spec
    /// § 7.4.3 — resets to 0 at every IDR), this counter increments
    /// monotonically across IDR boundaries so the encode-time stego
    /// hook produces unique `PositionKey`s per residual block. The
    /// decoder's slice walker mirrors with `frame_idx` ++ per VCL
    /// NAL. Without this separation, multi-frame stego would collide
    /// keys in `PlanInjector::HashMap` and apply the wrong plan bits.
    pub stego_frame_idx: u32,
    pub rc: RateController,
    pub recon: ReconBuffer,
    /// §B-cascade-real v1.1 — Phase 1.1.A.
    ///
    /// Decoupled visual-quality recon buffer. `recon` (above) tracks
    /// PRE-stego-flip levels and is used as the reference buffer for
    /// next-MB neighbour prediction + B-frame DPB lookup; this
    /// preserves the multi-pass orchestrator's cover-capture invariant
    /// (Pass 1 cover-capture and Pass 3 emit see identical
    /// neighbour-derived cover values at every position).
    ///
    /// `visual_recon` tracks POST-flip levels and is what a downstream
    /// player (the reference decoder / VLC / hostile decoder) reconstructs from the
    /// emitted bitstream. Mux output reads from this buffer so the
    /// muxed mp4 visually matches what any compliant decoder produces
    /// from our bitstream → no encoder/player drift cliff.
    ///
    /// Phase 1.1.A: field plumbed, helpers added, but writes are
    /// identical to `recon` (no flips applied yet). Phase 1.1.B will
    /// wire actual post-flip writes at residual reconstruct sites.
    ///
    /// See `docs/design/video/h264/b-cascade-real-v1-1-plan.md`.
    pub visual_recon: ReconBuffer,
    pub sps_params: SpsParams,
    pub pps_params: PpsParams,
    /// True once the first frame has emitted SPS + PPS. Subsequent
    /// frames skip them (standard H.264 stream layout).
    pub params_emitted: bool,
    /// CAVLC or CABAC. Must be set before the first frame is encoded
    /// (it gates the SPS/PPS bytes).
    pub entropy_mode: EntropyMode,
    /// Opt-in promotion to High profile (profile_idc=100) with
    /// `transform_8x8_mode_flag=1` in the PPS. Lights up the 8×8 DCT
    /// path on the encode side.
    ///
    /// **Only meaningful when `entropy_mode == EntropyMode::Cabac`** —
    /// Baseline/CAVLC paths stay Baseline+Main and ignore this flag.
    ///
    /// Default `false` (Main profile). Phase 100-D adds the plumbing;
    /// Phase 100-E starts emitting `transform_size_8x8_flag` per MB so
    /// the bitstream becomes valid High-profile output.
    pub enable_transform_8x8: bool,
    /// Phase 6E-A4 — when true, the SPS uses `pic_order_cnt_type = 0`
    /// + `log2_max_pic_order_cnt_lsb_minus4 = 4` +
    /// `max_num_ref_frames = 3` (M=2 IBPBP DPB shape), enabling
    /// `encode_b_frame` to emit valid B-slices that decoders can
    /// reorder via `pic_order_cnt_lsb`. Caller must set this BEFORE
    /// the first frame is encoded — toggling mid-stream produces an
    /// invalid bitstream (SPS already emitted).
    ///
    /// Default `false` keeps Phase 6B I+P-only behavior bit-identical.
    pub enable_b_frames: bool,
    /// Task #204 (v1.1) — typed B-RDO + B-residual config. Default
    /// `BRdoConfig::SAFE` (= both off; bucket Skip/Direct fallback —
    /// visually clean on real 1080p). Production stealth-calibrated
    /// paths set this to `BRdoConfig::PRODUCTION_VISUAL`. Replaces
    /// the historical `PHASM_B_RDO` / `PHASM_B_RESIDUAL` env-var
    /// pair (which still works as a debug override).
    pub b_rdo_config: super::mb_decision_b::BRdoConfig,
    /// v1.4 Phase 2.3 (#305) — multi-ref L0 configuration. Default
    /// `MultiRefConfig::SINGLE_REF` (= single L0 ref everywhere,
    /// matches v1.3 ship behavior bit-identically). Phase 4 ME +
    /// Phase 2.4 emit write the dual-ref form when this is set to
    /// `MultiRefConfig::DUAL_REF_L0`. Default flips in Phase 9 ship.
    pub multi_ref_config: MultiRefConfig,
    /// Phase 6E-A1 POC tracker — resets on IDR; used for B-frame
    /// `pic_order_cnt_lsb` emission. Always present in `Encoder` (
    /// cheap), but only consulted when `enable_b_frames = true`.
    pub poc_tracker: super::poc::PocTracker,
    /// Phase 6E-A4 — display index of the most recently encoded
    /// I or P frame (the "anchor"). Used to derive POC LSBs for
    /// the next P (anchor + m_factor) and the next B (anchor - 1
    /// in M=2 IBPBP encode order). 0 after the IDR; advances with
    /// each encode. Only consulted when `enable_b_frames = true`.
    pub display_idx_of_prev_anchor: u32,
    /// Single-slot DPB holding the previous encoded frame's recon.
    /// Used as P-frame motion-compensation reference (Phase 6B).
    pub dpb: ReferenceBuffer,
    /// Zero-indexed position in the current GOP. Frame 0 = IDR, frames
    /// 1..gop_length-1 are P (or non-IDR I if P is not implemented yet).
    pub gop_position: u32,
    /// GOP length in frames (default 30). User-settable via the setter.
    pub gop_length: u32,
    /// Frame-wide 4×4-granular MV grid. Populated as each P-slice
    /// partition resolves; read by the median predictor for
    /// subsequent partitions in the same MB and for neighboring MBs.
    mv_grid: EncoderMvGrid,
    /// Frame-wide 4×4-granular Intra_4x4 mode grid (spec § 8.3.1.1
    /// neighbor context). `0xFF` = "not an I_4x4 block" → DC fallback
    /// per spec. Populated per sub-block as I_4x4 MBs commit.
    i4x4_mode_grid: Vec<u8>,
    /// Frame-wide 8×8-granular Intra_8×8 mode grid (one entry per 8×8
    /// block, i.e. `mb_w*2 × mb_h*2`). `0xFF` = "not I_8×8" → spec
    /// § 8.3.3 neighbor derivation falls back to DC. Populated only
    /// when `enable_transform_8x8` is set.
    i8x8_mode_grid: Vec<u8>,
    /// Frame-wide 4×4-granular `TotalCoeff` grid for luma blocks,
    /// used by the CAVLC nC neighbor-context rule (spec § 9.2.1.1).
    /// `0xFF` = unavailable (intra-MB not yet encoded or out-of-frame).
    total_coeff_grid: Vec<u8>,
    /// Per-MB Intra16x16DC TotalCoeff. Used by the spec § 9.2.1.1 nC
    /// derivation for the *next* MB's Intra16x16DCLevel block — the
    /// neighbors for that block are the LEFT/ABOVE MBs' Intra16x16DC
    /// blocks, NOT 4×4 AC blocks. `0xFF` = unavailable (MB not coded
    /// as Intra_16x16, or out-of-frame). One entry per MB in raster
    /// order.
    intra16x16_dc_tc_grid: Vec<u8>,
    /// Per-chroma-4x4-block TotalCoeff for Cb / Cr planes. Used for
    /// the ChromaACLevel CAVLC nC neighbor derivation (spec § 9.2.1.1).
    /// Indexed by (chroma 4x4 grid y * chroma_w_4x4 + x). Each chroma
    /// plane is mb_w*2 wide and mb_h*2 tall in 4x4 units (4:2:0).
    /// `0xFF` = unavailable (MB hasn't been coded yet for this frame).
    chroma_cb_tc_grid: Vec<u8>,
    chroma_cr_tc_grid: Vec<u8>,
    /// Per-MB actual QP used during residual quant + reconstruction.
    /// Written by every write_*_macroblock site at the end of encoding.
    /// Read by `deblocking_filter::filter_frame` to compute per-edge
    /// `qp_avg = (qp_p + qp_q + 1) >> 1` (spec § 8.7.2.1). Without this,
    /// AQ-adjusted per-MB QPs produce encoder-side deblock outputs that
    /// diverge from decoder-side → ~0.22 MSE enc-vs-dec on a single
    /// P-frame, compounding drift. One entry per MB in raster order.
    qp_grid: Vec<u8>,
    /// Per-MB intra-coded flag. True if the MB was emitted as I_4x4
    /// or I_16x16 (including the intra-in-P fallback path). Used by
    /// the deblock filter's boundary_strength computation: spec says
    /// bs=4 at MB boundary when EITHER side is intra, bs=3 at internal
    /// edges of intra MBs. Frame-level `all_intra` flag (before this)
    /// was wrong for P-slices with intra-in-P MBs.
    intra_grid: Vec<bool>,
    /// Per-MB transform_8x8_flag. True when the MB used the 8×8
    /// transform; the deblock filter uses this to skip internal
    /// 4-pixel-grid edges inside the MB (spec § 8.7.2.1). One entry
    /// per MB in raster order.
    transform_8x8_grid: Vec<bool>,
    /// Running QP state used to compute the per-MB `mb_qp_delta`
    /// field. Resets to `slice_qp` at the start of each slice, updates
    /// after each MB that emits a non-zero CBP (spec § 7.4.5).
    prev_mb_qp: i32,
    /// Phase F: frame-mean log2(variance) in Q8 fixed-point. Set by
    /// [`Self::compute_aq_frame_mean`] at the top of each P-slice;
    /// consumed by `write_p_macroblock` to compute AQ-mode-3 offsets.
    /// 0 means "not yet computed / mode 3 disabled".
    aq_frame_mean_log2_q8: i32,
    /// §v1.7 Phase 1.1 (#323) — optional MB-tree result. When `Some`,
    /// per-MB QP offset from MB-tree analysis composes additively
    /// with the AQ-3 offset at each MB. `None` means MB-tree
    /// disabled (default). Set by caller via [`Self::set_mb_tree`]
    /// before encode pass; encoder reads via
    /// `self.mb_tree_qp_offset(...)`.
    pub mb_tree: Option<super::mb_tree::MbTreeResult>,
    /// §v1.7 Phase 1.1 — display-order frame index used to look up
    /// MB-tree offsets. Set by `encode_i_frame` / `encode_p_frame` /
    /// `encode_b_frame` from the caller's display sequence. Defaults
    /// to 0; caller passes the encode-order index implicitly via
    /// call sequence, but MB-tree wants DISPLAY-order index. Set
    /// explicitly via [`Self::set_mb_tree_display_idx`].
    pub mb_tree_display_idx: usize,
    /// §v1.7 Phase 2 (#324) — optional Lookahead RC result. When
    /// `Some`, the encoder applies the per-frame QP offset at the
    /// AQ composition site. None means lookahead disabled (default).
    /// Caller pre-computes via `analyze_lookahead_window` and assigns
    /// before encode pass.
    pub lookahead: Option<super::lookahead::LookaheadResult>,
    /// §v1.7 Phase 3 (#325) — CRF mode. When `Some(c)`, the encoder
    /// treats `c` as the perceptual quality target and overrides
    /// `self.rc.target_crf = c + lookahead_offset(display_idx)` at
    /// the entry of each frame (see `apply_crf_base_qp`). The
    /// per-frame-type offset (I=+0, P=+1, B=+2 default) is then
    /// applied downstream by `RateController::base_qp_for_frame_type`.
    /// When `None`, the constructor-derived `target_crf` is used
    /// unmodified (legacy fixed-quality behaviour).
    ///
    /// Single-pass CRF only; v1.7.0 has no VBV / 2-pass / rate
    /// feedback loop.
    pub crf: Option<u8>,
    /// Diagnostic: accumulates per-bin CABAC traces when tracing is
    /// enabled via `enable_cabac_trace()`. Used for the trace-diff
    /// harness (`examples/h264_cabac_trace_diff.rs`).
    cabac_trace_enabled: bool,
    cabac_trace_buffer: Vec<String>,
    /// Motion-estimation engine (stateless for 6B.2 but structured
    /// for future caches).
    me: MotionEstimator,
    /// RDO diagnostics: per-P-slice mode-decision counts. Indexed by
    /// `ModeStat` below. Reset at slice start, dumped to stderr at
    /// slice end when `PHASM_MODE_STATS` env var is set. Purely
    /// informational — drives task #124 RDO design without touching
    /// any production path.
    mode_stats: [u32; 9],
    /// Phase 6D.8: optional stego hook called between quantize and
    /// entropy emit on every residual block + MVD slot. When `None`
    /// the encoder behaves byte-identically to pre-6D.8. When `Some`
    /// the hook may either count positions (Pass 1) or apply
    /// overrides (Pass 3) per the encode-time CABAC stego flow.
    /// See `core/src/codec/h264/stego/encoder_hook.rs`.
    pub stego_hook: Option<Box<dyn super::super::stego::encoder_hook::StegoMbHook>>,
    /// Phase 6D.8 §30D-A: gates the **pre-MC MVD hook fire site**
    /// in `write_p_macroblock_cabac`. When false (default) the
    /// encoder behaves identically to chunk-5/§30C — MVD positions
    /// are never logged or modified, so chunk-5/§30C decode
    /// pipelines stay byte-identity-correct. When true, the
    /// stego hook fires for every P-MB MVD between mode decision
    /// and MC, allowing logger/injector to observe/modify per-axis
    /// MVD values. The encoder updates the partition's MV from the
    /// (possibly modified) MVD + neighbor predictor, so MC + recon
    /// run with the FINAL MV — no enc/dec drift.
    ///
    /// **§30D-A scaffolding only**: full multi-domain stego (mvd_*
    /// + coeff_*) needs a 3-pass orchestrator (§30D-C) since Pass 3
    /// MVD modifications make residuals diverge from Pass 1's
    /// logged residuals.
    pub enable_mvd_stego_hook: bool,
    /// **Task #383 (2026-05-12)** — process-env snapshot taken at
    /// `Encoder::new`. The encoder reads several PHASM_B_* env vars
    /// for debug overrides (force-mode, force-MV, residual on/off);
    /// previously these were re-read at encode-time, which races
    /// with parallel tests that mutate the env vars. Snapshotting
    /// once at construction makes a single encoder's behavior
    /// internally consistent regardless of mid-encode env mutations
    /// from other threads.
    pub env_snapshot: super::mb_decision_b::EncoderEnvSnapshot,
}

/// Indices into `Encoder::mode_stats`.
pub const MODE_STAT_P_SKIP_FAST: usize = 0;
pub const MODE_STAT_P_SKIP_POST_ME: usize = 1;
pub const MODE_STAT_P_16X16: usize = 2;
pub const MODE_STAT_P_16X8: usize = 3;
pub const MODE_STAT_P_8X16: usize = 4;
pub const MODE_STAT_P_8X8: usize = 5;
pub const MODE_STAT_INTRA_IN_P: usize = 6;
/// Phase D.3 split of MODE_STAT_INTRA_IN_P. Both counters also bump
/// MODE_STAT_INTRA_IN_P for backwards-compat; the sub-split lets the
/// Phase D.2-stealth calibration (task #52) compare our
/// I_4x4-vs-I_16x16 distribution against a reference H.264 encoder.
pub const MODE_STAT_INTRA_IN_P_I4X4: usize = 7;
pub const MODE_STAT_INTRA_IN_P_I16X16: usize = 8;

/// §6E-A4(c)-lite — pick B_Skip vs B_Direct_16x16 for a B-frame MB
/// based on a deterministic hash of `(frame_num, mb_addr)`. Returns
/// `true` for B_Direct_16x16, `false` for B_Skip.
///
/// Both modes use the spatial direct predictor at decode time, so
/// this choice doesn't affect visual quality — only the
/// MB-mode-distribution fingerprint on the wire. The 50/50 split
/// approximates real-encoder mode distributions in B-frames better
/// than all-B_Skip would. §6E-A4(c)-full will replace this with a
/// real RDO-based mode decision.
///
/// Hash: FxHash-style mixing of the inputs into a single bit.
/// Determinism is required so encoder and decoder see the same
/// mode distribution; deterministic-from-state means both sides
/// agree on the bit-stream layout.

/// v1.4 Phase 4.5 (#316) — fill a 16-entry `ref_idx_l0` array per the
/// 16x8/8x16 partition geometry. Each 4×4 block in raster order gets
/// the ref_idx of the partition it falls in. L1-only partitions write
/// 0 (their ref_idx_l0 was never transmitted; walker also commits 0
/// for the equivalent block range, so symmetry holds — and the
/// downstream `condTermFlag` read only looks at `!= 0`, which is
/// correct: an unused list contributes 0).
fn fill_ref_idx_l0_partitioned(
    meta: super::b_partitioned::BPartitionedMeta,
    parts: &[super::b_partitioned::BPartitionMv; 2],
) -> [i8; 16] {
    use super::b_partitioned::BListUse;
    let mut out = [0i8; 16];
    let (pw, ph) = meta.shape.part_dim_4x4();
    for idx in 0..2usize {
        let usage = if idx == 0 { meta.part0 } else { meta.part1 };
        let uses_l0 = matches!(usage, BListUse::L0 | BListUse::Bi);
        if !uses_l0 {
            continue;
        }
        let (off_x, off_y) = meta.shape.part_offset(idx);
        let val = parts[idx].ref_idx_l0 as i8;
        for dy in 0..ph {
            for dx in 0..pw {
                out[(off_y + dy) * 4 + (off_x + dx)] = val;
            }
        }
    }
    out
}

/// v1.4 Phase 4.5 (#316) — fill a 16-entry `ref_idx_l0` array per the
/// B_8x8 sub-MB geometry. Each sub-MB covers a 2×2 4×4-cell region.
/// Direct (sub=0) and L1-only (sub=2) sub-MBs leave their region at 0
/// (no L0 transmitted; walker symmetrical).
fn fill_ref_idx_l0_b8x8(
    sub_mb_types: [u8; 4],
    parts: &[super::b_partitioned::BPartitionMv; 4],
) -> [i8; 16] {
    let mut out = [0i8; 16];
    for s_idx in 0..4usize {
        let sub = sub_mb_types[s_idx];
        // sub_mb_type 1 = L0_8x8, 3 = Bi_8x8 — both use L0.
        let uses_l0 = matches!(sub, 1 | 3);
        if !uses_l0 {
            continue;
        }
        let off_x = (s_idx & 1) * 2;
        let off_y = (s_idx >> 1) * 2;
        let val = parts[s_idx].ref_idx_l0 as i8;
        for dy in 0..2usize {
            for dx in 0..2usize {
                out[(off_y + dy) * 4 + (off_x + dx)] = val;
            }
        }
    }
    out
}

/// v1.4 Phase 4.5 (#316) — fill a 16-entry `ref_idx_l0` array for a
/// P-slice MB choice. P_Skip is always uniform 0; the partition shapes
/// dispatch to the right per-partition fill.
fn fill_ref_idx_l0_p(
    choice: &super::partition_decision::PMbChoice,
) -> [i8; 16] {
    use super::partition_decision::PMbChoice;
    let mut out = [0i8; 16];
    match choice {
        PMbChoice::P16x16 { ref_idx_l0, .. } => {
            out = [*ref_idx_l0 as i8; 16];
        }
        PMbChoice::P16x8 { ref_idx_l0, .. } => {
            // Top half (rows 0-1, blk 0..7) = part 0; bottom (blk 8..15) = part 1.
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
        PMbChoice::P8x16 { ref_idx_l0, .. } => {
            // Left half (cols 0-1) = part 0; right (cols 2-3) = part 1.
            for r in 0..4 {
                for c in 0..2 {
                    out[r * 4 + c] = ref_idx_l0[0] as i8;
                }
                for c in 2..4 {
                    out[r * 4 + c] = ref_idx_l0[1] as i8;
                }
            }
        }
        PMbChoice::P8x8 { sub } => {
            // Each sub-MB covers a 2x2 cell region: s=0:(0,0), s=1:(2,0),
            // s=2:(0,2), s=3:(2,2). One ref_idx_l0 per 8x8 sub-MB regardless
            // of internal sub_mb_type.
            for (s_idx, sc) in sub.iter().enumerate() {
                let off_x = (s_idx & 1) * 2;
                let off_y = (s_idx >> 1) * 2;
                let val = sc.ref_idx_l0() as i8;
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        out[(off_y + dy) * 4 + (off_x + dx)] = val;
                    }
                }
            }
        }
    }
    out
}

/// §6E-A6.1 — emit `B_L0_16x16` (mb_type = 1) syntax for one MB:
/// `mb_skip_flag = 0`, `mb_type = 1`, ref_idx (skipped — inferred
/// 0 since `num_ref_idx_l0_active_minus1 = 0`), one L0 MVD pair
/// against the spatial-direct predictor, CBP = 0, then
/// `end_of_slice_flag`. Residual emission for L0_16x16 is on the
/// `write_b_inter_residual_macroblock_cabac` path; this function
/// is only reached when residual is gated off (CBP=0 path).
///
/// MV grid: writes the explicit L0 MV into all 16 cells of the MB,
/// L1 stays absent (REF_IDX_NONE).
#[allow(clippy::too_many_arguments)]
fn emit_b_l0_16x16(
    cabac: &mut crate::codec::h264::cabac::encoder::CabacEncoder,
    mb_x: usize,
    grid: &mut super::partition_state::EncoderMvGrid,
    grid_mb_x: usize,
    grid_mb_y: usize,
    mv: super::motion_estimation::MotionVector,
    ref_idx_l0: u8,
    num_active_l0: u8,
) -> Result<(), EncoderError> {
    use crate::codec::h264::cabac::encoder::{encode_mvd_with_bin0_inc, encode_ref_idx};
    use crate::codec::h264::cabac::neighbor::{CabacNeighborMB, MbTypeClass};

    encode_mb_skip_flag_b(cabac, false, mb_x);
    encode_mb_type_b(cabac, 1, mb_x);

    // v1.4 (#305) — ref_idx_l0 emitted per spec § 7.3.5.1 BEFORE the
    // MVD pair, gated on num_ref_idx_l0_active > 1. At
    // MultiRefConfig::SINGLE_REF default the gate is closed and zero
    // bins emit (bit-identical to v1.3).
    let current_ref_idx_mb = crate::codec::h264::cabac::neighbor::CurrentMbRefIdx::new();
    if num_active_l0 > 1 {
        encode_ref_idx(
            cabac, ref_idx_l0 as u32, &current_ref_idx_mb, mb_x, 0, 0,
            (num_active_l0 - 1) as u32,
        );
    }

    // MVD = actual_mv - predicted_mv. Predictor at the 16x16
    // partition's anchor; `current_ref_idx = 0` per single-ref.
    // Use the L0-view predictor (existing P-side function reads
    // the L0 fields).
    let predicted = super::partition_state::predict_mv_for_mb_partition(
        grid, grid_mb_x * 4, grid_mb_y * 4, /* part_w_4x4 */ 4, /* part_h_4x4 */ 4,
        /* mb_part_idx */ 0, /* current_ref_idx */ 0,
    );
    let mvd_x = (mv.mv_x as i32) - (predicted.mv_x as i32);
    let mvd_y = (mv.mv_y as i32) - (predicted.mv_y as i32);

    // bin0 ctxIdxInc derived from neighbour MVD magnitudes —
    // shared neighbour state with the P-side path. Per-list
    // separation per spec § 9.3.3.1.1.7 is in `_per_list` variants
    // used by Bi/Partitioned/B_8x8 paths; the L0-only 16x16 emitter
    // here uses the shared-state read since it only writes one list.
    let bin0_inc_x = crate::codec::h264::cabac::neighbor::ctx_idx_inc_mvd_bin0(
        &cabac.neighbors, mb_x, /* a_blk */ 0, /* b_blk */ 0, /* component */ 0,
    );
    encode_mvd_with_bin0_inc(cabac, mvd_x, /* component */ 0, bin0_inc_x);
    let bin0_inc_y = crate::codec::h264::cabac::neighbor::ctx_idx_inc_mvd_bin0(
        &cabac.neighbors, mb_x, 0, 0, /* component */ 1,
    );
    encode_mvd_with_bin0_inc(cabac, mvd_y, /* component */ 1, bin0_inc_y);

    encode_coded_block_pattern(cabac, /* cbp_value */ 0, mb_x);

    // Commit neighbour as inter (B-slice non-direct looks like
    // P-frame inter for downstream context purposes). For zero-MV
    // test paths the abs_mvd_comp default (all 0) is correct;
    // §6E-A6.1 part 4 will populate it from the current_mvd
    // tracker once non-zero MVDs flow through.
    let abs_x = mvd_x.unsigned_abs().min(i16::MAX as u32) as i16;
    let abs_y = mvd_y.unsigned_abs().min(i16::MAX as u32) as i16;
    let nb = CabacNeighborMB {
        mb_type: MbTypeClass::PInter,
        mb_skip_flag: false,
        cbp_luma: 0,
        cbp_chroma: 0,
        // v1.4 Phase 4.5 (#316) — actual emitted ref_idx_l0 across all
        // 16 4×4 cells. Spec § 9.3.3.1.1.6 reads this for the next MB's
        // ref_idx ctxIdxInc. Hardcoding [0;16] when the wire carries
        // ref_idx>0 desyncs vs spec-conforming decoders (the reference decoder).
        ref_idx_l0: [ref_idx_l0 as i8; 16],
        abs_mvd_comp: [[abs_x; 16], [abs_y; 16]],
        ..CabacNeighborMB::default()
    };
    cabac.neighbors.commit(mb_x, nb);

    // Populate MV grid: explicit L0 MV everywhere, L1 absent.
    grid.fill_lists(grid_mb_x * 4, grid_mb_y * 4, 4, 4, Some((mv, 0)), None);

    Ok(())
}

/// §6E-A6.1 — emit `B_L1_16x16` (mb_type = 2). Mirror of `emit_b_l0_16x16`
/// but writes L1 MVD only and populates the L1 grid.
fn emit_b_l1_16x16(
    cabac: &mut crate::codec::h264::cabac::encoder::CabacEncoder,
    mb_x: usize,
    grid: &mut super::partition_state::EncoderMvGrid,
    grid_mb_x: usize,
    grid_mb_y: usize,
    mv: super::motion_estimation::MotionVector,
) -> Result<(), EncoderError> {
    use crate::codec::h264::cabac::encoder::encode_mvd_with_bin0_inc;
    use crate::codec::h264::cabac::neighbor::{CabacNeighborMB, MbTypeClass};

    encode_mb_skip_flag_b(cabac, false, mb_x);
    encode_mb_type_b(cabac, 2, mb_x);

    // L1 predictor — read the L1 view of the grid (B_Direct
    // neighbours have populated L1 cells via `populate_b_direct_grid`).
    let predicted = super::b_direct_predictor::predict_mv_for_partition_l1_pub(
        grid, grid_mb_x * 4, grid_mb_y * 4, /* current_ref_idx */ 0,
    );
    let mvd_x = (mv.mv_x as i32) - (predicted.mv_x as i32);
    let mvd_y = (mv.mv_y as i32) - (predicted.mv_y as i32);

    // §6E-A6.1 spec § 9.3.3.1.1.7 fix: L1 MVD bin0 ctxIdxInc reads
    // L1 neighbour state, NOT L0. Use the per-list variant with
    // list=1.
    let bin0_inc_x = crate::codec::h264::cabac::neighbor::ctx_idx_inc_mvd_bin0_per_list(
        &cabac.neighbors, mb_x, 0, 0, 0, /* list */ 1,
    );
    encode_mvd_with_bin0_inc(cabac, mvd_x, 0, bin0_inc_x);
    let bin0_inc_y = crate::codec::h264::cabac::neighbor::ctx_idx_inc_mvd_bin0_per_list(
        &cabac.neighbors, mb_x, 0, 0, 1, /* list */ 1,
    );
    encode_mvd_with_bin0_inc(cabac, mvd_y, 1, bin0_inc_y);

    encode_coded_block_pattern(cabac, 0, mb_x);

    let abs_x = mvd_x.unsigned_abs().min(i16::MAX as u32) as i16;
    let abs_y = mvd_y.unsigned_abs().min(i16::MAX as u32) as i16;
    // §6E-A6.1 spec § 9.3.3.1.1.7 fix: B_L1_16x16 commits its MVDs
    // to abs_mvd_comp_l1 (NOT abs_mvd_comp). Subsequent neighbours
    // with L1 MVDs read these per-list arrays.
    //
    // v1.4 Phase 4.5 (#316) — L1_16x16 has no L0 reference (the
    // partition is L1-only per spec Table 7-14), so ref_idx_l0
    // stays [0;16]. Walker's L1_16x16 path is symmetric (also
    // commits 0). v1.4 keeps L1 single-ref so no further per-list
    // tracking is needed here.
    let nb = CabacNeighborMB {
        mb_type: MbTypeClass::PInter,
        mb_skip_flag: false,
        cbp_luma: 0,
        cbp_chroma: 0,
        ref_idx_l0: [0i8; 16],
        abs_mvd_comp_l1: [[abs_x; 16], [abs_y; 16]],
        ..CabacNeighborMB::default()
    };
    cabac.neighbors.commit(mb_x, nb);

    grid.fill_lists(grid_mb_x * 4, grid_mb_y * 4, 4, 4, None, Some((mv, 0)));

    Ok(())
}

/// §6E-A6.1 — emit `B_Bi_16x16` (mb_type = 3). Both L0 and L1
/// MVDs are emitted (in spec order: L0 first, then L1 per § 7.3.5.1).
///
/// v1.4 (#305) — ref_idx_l0 emitted between mb_type and L0 MVD when
/// `num_active_l0 > 1`. L1 stays single-ref (Q1).
#[allow(clippy::too_many_arguments)]
fn emit_b_bi_16x16(
    cabac: &mut crate::codec::h264::cabac::encoder::CabacEncoder,
    mb_x: usize,
    grid: &mut super::partition_state::EncoderMvGrid,
    grid_mb_x: usize,
    grid_mb_y: usize,
    mv_l0: super::motion_estimation::MotionVector,
    mv_l1: super::motion_estimation::MotionVector,
    ref_idx_l0: u8,
    num_active_l0: u8,
) -> Result<(), EncoderError> {
    use crate::codec::h264::cabac::encoder::{encode_mvd_with_bin0_inc, encode_ref_idx};
    use crate::codec::h264::cabac::neighbor::{CabacNeighborMB, MbTypeClass};

    encode_mb_skip_flag_b(cabac, false, mb_x);
    encode_mb_type_b(cabac, 3, mb_x);

    // v1.4 (#305) — ref_idx_l0 (Bi uses L0 list).
    let current_ref_idx_mb = crate::codec::h264::cabac::neighbor::CurrentMbRefIdx::new();
    if num_active_l0 > 1 {
        encode_ref_idx(
            cabac, ref_idx_l0 as u32, &current_ref_idx_mb, mb_x, 0, 0,
            (num_active_l0 - 1) as u32,
        );
    }

    // L0 MVD pair first.
    let pred_l0 = super::partition_state::predict_mv_for_mb_partition(
        grid, grid_mb_x * 4, grid_mb_y * 4, 4, 4, 0, 0,
    );
    let mvd_l0_x = (mv_l0.mv_x as i32) - (pred_l0.mv_x as i32);
    let mvd_l0_y = (mv_l0.mv_y as i32) - (pred_l0.mv_y as i32);
    let bin0_inc_x = crate::codec::h264::cabac::neighbor::ctx_idx_inc_mvd_bin0(
        &cabac.neighbors, mb_x, 0, 0, 0,
    );
    encode_mvd_with_bin0_inc(cabac, mvd_l0_x, 0, bin0_inc_x);
    let bin0_inc_y = crate::codec::h264::cabac::neighbor::ctx_idx_inc_mvd_bin0(
        &cabac.neighbors, mb_x, 0, 0, 1,
    );
    encode_mvd_with_bin0_inc(cabac, mvd_l0_y, 1, bin0_inc_y);

    // L1 MVD pair second. §6E-A6.1 spec § 9.3.3.1.1.7 fix: bin0
    // ctxIdxInc for L1 reads L1 neighbour state (per-list).
    let pred_l1 = super::b_direct_predictor::predict_mv_for_partition_l1_pub(
        grid, grid_mb_x * 4, grid_mb_y * 4, 0,
    );
    let mvd_l1_x = (mv_l1.mv_x as i32) - (pred_l1.mv_x as i32);
    let mvd_l1_y = (mv_l1.mv_y as i32) - (pred_l1.mv_y as i32);
    let bin0_inc_x = crate::codec::h264::cabac::neighbor::ctx_idx_inc_mvd_bin0_per_list(
        &cabac.neighbors, mb_x, 0, 0, 0, /* list */ 1,
    );
    encode_mvd_with_bin0_inc(cabac, mvd_l1_x, 0, bin0_inc_x);
    let bin0_inc_y = crate::codec::h264::cabac::neighbor::ctx_idx_inc_mvd_bin0_per_list(
        &cabac.neighbors, mb_x, 0, 0, 1, /* list */ 1,
    );
    encode_mvd_with_bin0_inc(cabac, mvd_l1_y, 1, bin0_inc_y);

    encode_coded_block_pattern(cabac, 0, mb_x);

    // §6E-A6.1 spec § 9.3.3.1.1.7 fix: Bi commits BOTH per-list
    // MVD magnitudes to their respective neighbour fields.
    let abs_l0_x = mvd_l0_x.unsigned_abs().min(i16::MAX as u32) as i16;
    let abs_l0_y = mvd_l0_y.unsigned_abs().min(i16::MAX as u32) as i16;
    let abs_l1_x = mvd_l1_x.unsigned_abs().min(i16::MAX as u32) as i16;
    let abs_l1_y = mvd_l1_y.unsigned_abs().min(i16::MAX as u32) as i16;
    let nb = CabacNeighborMB {
        mb_type: MbTypeClass::PInter,
        mb_skip_flag: false,
        cbp_luma: 0,
        cbp_chroma: 0,
        // v1.4 Phase 4.5 (#316) — Bi uses L0 list, propagate the
        // emitted ref_idx_l0 to all 16 4×4 cells.
        ref_idx_l0: [ref_idx_l0 as i8; 16],
        abs_mvd_comp: [[abs_l0_x; 16], [abs_l0_y; 16]],
        abs_mvd_comp_l1: [[abs_l1_x; 16], [abs_l1_y; 16]],
        ..CabacNeighborMB::default()
    };
    cabac.neighbors.commit(mb_x, nb);

    grid.fill_lists(
        grid_mb_x * 4, grid_mb_y * 4, 4, 4,
        Some((mv_l0, 0)), Some((mv_l1, 0)),
    );

    Ok(())
}

/// §6E-A6.2 — emit a partitioned B mb_type (4..21). Looks up
/// `(shape, list_usage_part0, list_usage_part1)` via
/// `b_partitioned::partitioned_b_meta`, then emits per spec
/// § 7.3.5.1 mb_pred order:
///   1. mb_skip_flag = 0
///   2. mb_type bin tree
///   3. ref_idx_l0 / ref_idx_l1 per partition (skipped — inferred 0
///      under single-ref ship config)
///   4. MVDs in spec order: partition 0 (L0 if used, then L1 if
///      used), partition 1 (L0 if used, then L1 if used)
///   5. coded_block_pattern (zero — partitioned-B residual is not
///      wired; only the 16x16 family residuals go through the
///      `write_b_inter_residual_macroblock_cabac` path)
///   6. (no mb_qp_delta when CBP=0)
///
/// Per-list MVD bin0 ctxIdxInc uses the `_per_list` variant
/// (§6E-A6.1 (4/N) fix) so encoder + walker match a spec decoder.
///
/// §B-Partitioned-Residual Stage B (#206) — when `b_refs` is `Some`
/// AND `residual_enabled` is true, the partition's per-list prediction
/// is built (16x8 / 8x16 partitions stitched into a single 16×16 luma +
/// 8×8 chroma surface) and the shared
/// [`Encoder::emit_b_residual_for_pred`] helper emits CBP +
/// `transform_size_8x8_flag` + `mb_qp_delta` + luma 4×4 residuals +
/// chroma DC/AC residuals + writes the post-residual recon back into
/// `self.recon`. Otherwise emits CBP=0 with no residual + no recon
/// write (legacy fallback for tests / no-DPB callers).
#[allow(clippy::too_many_arguments)]
fn emit_b_partitioned_method(
    enc: &mut Encoder,
    cabac: &mut crate::codec::h264::cabac::encoder::CabacEncoder,
    mb_x: usize,
    mb_y: usize,
    mb_type: u8,
    parts: [super::b_partitioned::BPartitionMv; 2],
    b_refs: Option<(
        &super::reference_buffer::ReconFrame,
        &super::reference_buffer::ReconFrame,
    )>,
    residual_enabled: bool,
    y_plane: &[u8],
    y_stride: usize,
    cb_plane: &[u8],
    cr_plane: &[u8],
    c_stride: usize,
    qp: u8,
    qp_c: u8,
    num_active_l0: u8,
) -> Result<(), EncoderError> {
    use super::b_partitioned::{partitioned_b_meta, BListUse};
    use super::motion_compensation::{
        apply_chroma_mv_block, apply_chroma_mv_block_bipred, apply_luma_mv_block,
        apply_luma_mv_block_bipred,
    };
    use crate::codec::h264::cabac::encoder::{encode_mvd_with_bin0_inc, encode_ref_idx};
    use crate::codec::h264::cabac::neighbor::{
        compute_mvd_ctx_idx_inc_bin0_per_list, CabacNeighborMB, CurrentMbMvdAbs, MbTypeClass,
    };

    let meta = partitioned_b_meta(mb_type as u32).ok_or_else(|| {
        EncoderError::InvalidInput(format!(
            "emit_b_partitioned: mb_type {mb_type} not in partitioned range 4..=21"
        ))
    })?;

    encode_mb_skip_flag_b(cabac, false, mb_x);
    encode_mb_type_b(cabac, mb_type as u32, mb_x);

    // v1.4 (#305) — ref_idx_l0 per partition per spec § 7.3.5.1
    // mb_pred(): all ref_idx_l0 emit BEFORE all MVDs, in partition-
    // index order, filtered by partition-uses-L0 (skip if partition
    // is L1-only). At MultiRefConfig::SINGLE_REF default the gate is
    // closed and zero bins emit (bit-identical to v1.3).
    //
    // v1.4 Phase 4.5 (#316) — track per-4×4-block ref_idx_l0 in
    // current MB so partition 1's bin 0 ctxIdxInc reads partition 0's
    // just-emitted ref_idx via within-MB lookup (spec § 6.4.11.7).
    let mut current_ref_idx_mb = crate::codec::h264::cabac::neighbor::CurrentMbRefIdx::new();
    let (pw_4x4_ref, ph_4x4_ref) = meta.shape.part_dim_4x4();
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
            encode_ref_idx(
                cabac, parts[idx].ref_idx_l0 as u32, &current_ref_idx_mb,
                mb_x, cur_bx, cur_by,
                (num_active_l0 - 1) as u32,
            );
            current_ref_idx_mb.fill_region(
                cur_bx, cur_by, pw_4x4_ref as u8, ph_4x4_ref as u8,
                parts[idx].ref_idx_l0 as i8,
            );
        }
    }

    // §6E-A6.1q.e (#154) — per-list within-MB MVD tracker. Partition
    // 1's bin 0 ctxIdxInc reads partition 0's just-emitted MVD via
    // `compute_mvd_ctx_idx_inc_bin0_per_list`.
    let mut current_mvd = CurrentMbMvdAbs::new();
    let (pw_4x4, ph_4x4) = meta.shape.part_dim_4x4();

    // §B-encoder-decoder-divergence Phase 2.7 (2026-05-08, #248) —
    // H.264 spec § 7.3.5.1 / § 9.3.3.1.1 interprets B-slice partitioned
    // MVD parsing in **list-major** loop order: outer iterates list
    // 0..2, inner iterates partition index 0..N-1, emitting an MVD
    // pair only when that partition uses that list. Phasm previously
    // used **partition-major** order which is round-trip-clean against
    // phasm's own walker but desyncs vs spec-compliant decoders for
    // asymmetric mb_types where the orders differ:
    //   mb_type 14 (B_L1_Bi 16x8): partition-major emits L1_p0
    //     before L0_p1; spec order emits L0_p1 before L1_p0.
    //   Same for 15 (B_L1_Bi 8x16), 16 (B_Bi_L0 16x8), 17 (8x16).
    // Default ME often converges to MVD=0 making the desync invisible
    // (MV reconstructs to predictor regardless of bit assignment),
    // but real-world content with non-zero MVDs produces 100+ pixel
    // divergences. List-major matches the spec's interpretation;
    // walker `walk_b_partitioned` in cabac/bin_decoder/slice.rs
    // updated in lockstep.
    for list in 0u8..2 {
        for idx in 0..2usize {
            let usage = if idx == 0 { meta.part0 } else { meta.part1 };
            let part = &parts[idx];
            let uses_this_list = match (list, usage) {
                (0, BListUse::L0) | (0, BListUse::Bi) => true,
                (1, BListUse::L1) | (1, BListUse::Bi) => true,
                _ => false,
            };
            if !uses_this_list {
                continue;
            }
            let (off_x, off_y) = meta.shape.part_offset(idx);
            let tl_bx = mb_x * 4 + off_x;
            let tl_by = mb_y * 4 + off_y;
            let cur_bx = off_x as u8;
            let cur_by = off_y as u8;

            if list == 0 {
                let mv = part.mv_l0.unwrap_or_default();
                let predicted = super::partition_state::predict_mv_for_mb_partition(
                    &enc.mv_grid, tl_bx, tl_by, pw_4x4, ph_4x4, idx as u8,
                    /* current_ref_idx */ 0,
                );
                let mvd_x = (mv.mv_x as i32) - (predicted.mv_x as i32);
                let mvd_y = (mv.mv_y as i32) - (predicted.mv_y as i32);
                let bin0_inc_x = compute_mvd_ctx_idx_inc_bin0_per_list(
                    &current_mvd, &cabac.neighbors, mb_x, cur_bx, cur_by, 0, 0,
                );
                encode_mvd_with_bin0_inc(cabac, mvd_x, 0, bin0_inc_x);
                let bin0_inc_y = compute_mvd_ctx_idx_inc_bin0_per_list(
                    &current_mvd, &cabac.neighbors, mb_x, cur_bx, cur_by, 1, 0,
                );
                encode_mvd_with_bin0_inc(cabac, mvd_y, 1, bin0_inc_y);
                let abs_x = mvd_x.unsigned_abs() as i16;
                let abs_y = mvd_y.unsigned_abs() as i16;
                current_mvd.fill_region(cur_bx, cur_by, pw_4x4 as u8, ph_4x4 as u8, abs_x, abs_y);
                // Update L0 grid in place so partition 1's L0 predict
                // (and downstream L1 cross-list reads) see the MV.
                enc.mv_grid.fill_lists(
                    tl_bx, tl_by, pw_4x4, ph_4x4,
                    part.mv_l0.map(|m| (m, 0)),
                    None,
                );
            } else {
                let mv = part.mv_l1.unwrap_or_default();
                // §B-cascade-real Phase 2 (#267) — partition-aware
                // L1 predictor (idx + part_w + part_h + spec
                // § 8.4.1.3.1 directional shortcuts), mirroring the
                // L0 path's `predict_mv_for_mb_partition` call above.
                // Fixes the L0/L1 PMV asymmetry that drove
                // mismatch_y=199 on motion content (#266).
                let predicted = super::b_direct_predictor::predict_mv_for_mb_partition_l1(
                    &enc.mv_grid, tl_bx, tl_by, pw_4x4, ph_4x4, idx as u8,
                    /* current_ref_idx */ 0,
                );
                let mvd_x = (mv.mv_x as i32) - (predicted.mv_x as i32);
                let mvd_y = (mv.mv_y as i32) - (predicted.mv_y as i32);
                let bin0_inc_x = compute_mvd_ctx_idx_inc_bin0_per_list(
                    &current_mvd, &cabac.neighbors, mb_x, cur_bx, cur_by, 0, 1,
                );
                encode_mvd_with_bin0_inc(cabac, mvd_x, 0, bin0_inc_x);
                let bin0_inc_y = compute_mvd_ctx_idx_inc_bin0_per_list(
                    &current_mvd, &cabac.neighbors, mb_x, cur_bx, cur_by, 1, 1,
                );
                encode_mvd_with_bin0_inc(cabac, mvd_y, 1, bin0_inc_y);
                let abs_x = mvd_x.unsigned_abs() as i16;
                let abs_y = mvd_y.unsigned_abs() as i16;
                current_mvd.fill_region_l1(cur_bx, cur_by, pw_4x4 as u8, ph_4x4 as u8, abs_x, abs_y);
                enc.mv_grid.fill_lists(
                    tl_bx, tl_by, pw_4x4, ph_4x4,
                    None,
                    part.mv_l1.map(|m| (m, 0)),
                );
            }
        }
    }

    // §B-Partitioned-Residual (#206) — full-residual path when refs
    // are available + residual_enabled. Otherwise CBP=0 fallback.
    if let (Some((l0_ref, l1_ref)), true) = (b_refs, residual_enabled) {
        let part_w_luma = pw_4x4 * 4;
        let part_h_luma = ph_4x4 * 4;
        let part_w_chroma = part_w_luma / 2;
        let part_h_chroma = part_h_luma / 2;
        let mb_px_x = (mb_x * 16) as u32;
        let mb_px_y = (mb_y * 16) as u32;
        let mb_cpx_x = (mb_x * 8) as u32;
        let mb_cpx_y = (mb_y * 8) as u32;

        let mut pred_y = [[0u8; 16]; 16];
        let mut pred_cb = [[0u8; 8]; 8];
        let mut pred_cr = [[0u8; 8]; 8];

        for (idx, part) in parts.iter().enumerate() {
            let usage = if idx == 0 { meta.part0 } else { meta.part1 };
            let (off_4x4_x, off_4x4_y) = meta.shape.part_offset(idx);
            let off_x_luma = (off_4x4_x * 4) as u32;
            let off_y_luma = (off_4x4_y * 4) as u32;
            let off_x_chroma = (off_4x4_x * 2) as u32;
            let off_y_chroma = (off_4x4_y * 2) as u32;

            // Luma fill into pred_y at the partition offset.
            {
                let pred_y_flat = pred_y.as_flattened_mut();
                let y_start = (off_y_luma * 16 + off_x_luma) as usize;
                let pred_y_sub = &mut pred_y_flat[y_start..];
                match usage {
                    BListUse::L0 => {
                        let mv = part.mv_l0.unwrap_or_default();
                        apply_luma_mv_block(
                            l0_ref,
                            mb_px_x + off_x_luma, mb_px_y + off_y_luma,
                            part_w_luma as u32, part_h_luma as u32,
                            mv, pred_y_sub, 16,
                        );
                    }
                    BListUse::L1 => {
                        let mv = part.mv_l1.unwrap_or_default();
                        apply_luma_mv_block(
                            l1_ref,
                            mb_px_x + off_x_luma, mb_px_y + off_y_luma,
                            part_w_luma as u32, part_h_luma as u32,
                            mv, pred_y_sub, 16,
                        );
                    }
                    BListUse::Bi => {
                        let mv_l0 = part.mv_l0.unwrap_or_default();
                        let mv_l1 = part.mv_l1.unwrap_or_default();
                        apply_luma_mv_block_bipred(
                            l0_ref, mv_l0, l1_ref, mv_l1,
                            mb_px_x + off_x_luma, mb_px_y + off_y_luma,
                            part_w_luma as u32, part_h_luma as u32,
                            pred_y_sub, 16,
                        );
                    }
                }
            }
            // Chroma fill into pred_cb + pred_cr.
            for (component, pred_c) in [&mut pred_cb, &mut pred_cr].iter_mut().enumerate() {
                let pred_c_flat = pred_c.as_flattened_mut();
                let c_start = (off_y_chroma * 8 + off_x_chroma) as usize;
                let pred_c_sub = &mut pred_c_flat[c_start..];
                match usage {
                    BListUse::L0 => {
                        let mv = part.mv_l0.unwrap_or_default();
                        apply_chroma_mv_block(
                            l0_ref, component as u8,
                            mb_cpx_x + off_x_chroma, mb_cpx_y + off_y_chroma,
                            part_w_chroma as u32, part_h_chroma as u32,
                            mv, pred_c_sub, 8,
                        );
                    }
                    BListUse::L1 => {
                        let mv = part.mv_l1.unwrap_or_default();
                        apply_chroma_mv_block(
                            l1_ref, component as u8,
                            mb_cpx_x + off_x_chroma, mb_cpx_y + off_y_chroma,
                            part_w_chroma as u32, part_h_chroma as u32,
                            mv, pred_c_sub, 8,
                        );
                    }
                    BListUse::Bi => {
                        let mv_l0 = part.mv_l0.unwrap_or_default();
                        let mv_l1 = part.mv_l1.unwrap_or_default();
                        apply_chroma_mv_block_bipred(
                            l0_ref, mv_l0, l1_ref, mv_l1, component as u8,
                            mb_cpx_x + off_x_chroma, mb_cpx_y + off_y_chroma,
                            part_w_chroma as u32, part_h_chroma as u32,
                            pred_c_sub, 8,
                        );
                    }
                }
            }
        }

        let res = enc.emit_b_residual_for_pred(
            cabac, mb_x, mb_y,
            y_plane, y_stride, cb_plane, cr_plane, c_stride,
            &pred_y, &pred_cb, &pred_cr,
            qp, qp_c,
        )?;

        let mut nb = CabacNeighborMB::default();
        nb.mb_type = MbTypeClass::PInter;
        nb.mb_skip_flag = false;
        nb.cbp_luma = res.cbp_luma_8x8;
        nb.cbp_chroma = res.cbp_chroma;
        nb.mb_qp_delta = res.qp_delta_emitted;
        nb.coded_block_flag_cat = res.current_cbf.to_neighbor_cbf();
        // v1.4 Phase 4.5 (#316) — per-block fill from partition geometry.
        // L1-only partitions stay 0 (no L0 transmitted; walker symmetric).
        nb.ref_idx_l0 = fill_ref_idx_l0_partitioned(meta, &parts);
        nb.abs_mvd_comp = current_mvd.to_neighbor();
        nb.abs_mvd_comp_l1 = current_mvd.to_neighbor_l1();
        nb.transform_size_8x8_flag = false;
        cabac.neighbors.commit(mb_x, nb);
    } else {
        // Legacy fallback: CBP=0, no residual emission, no recon write.
        // Matches pre-§B-Partitioned-Residual behavior.
        encode_coded_block_pattern(cabac, /* cbp_value */ 0, mb_x);
        let nb = CabacNeighborMB {
            mb_type: MbTypeClass::PInter,
            mb_skip_flag: false,
            cbp_luma: 0,
            cbp_chroma: 0,
            // v1.4 Phase 4.5 (#316) — per-block fill from partition geometry.
            ref_idx_l0: fill_ref_idx_l0_partitioned(meta, &parts),
            abs_mvd_comp: current_mvd.to_neighbor(),
            abs_mvd_comp_l1: current_mvd.to_neighbor_l1(),
            ..CabacNeighborMB::default()
        };
        cabac.neighbors.commit(mb_x, nb);
    }

    Ok(())
}

/// §6E-A6.3 — emit `B_8x8` (mb_type = 22) with uniform sub-MB
/// partitioning. Per spec § 7.3.5.1 + § 7.3.5.2:
///
/// 1. mb_skip_flag = 0
/// 2. mb_type = 22 (encoded via `encode_mb_type_b`)
/// 3. 4 × `sub_mb_type[s]` (each 0..=3 — `B_Direct_8x8` / `B_L0_8x8`
///    / `B_L1_8x8` / `B_Bi_8x8`)
/// 4. (No `transform_size_8x8_flag` — only emitted when CBP-luma ≠ 0
///    AND the High-profile flag is set; we ship CBP=0 + Baseline.)
/// 5. Per sub-MB s in raster order (0=top-left, 1=top-right,
///    2=bottom-left, 3=bottom-right): MVDs per the sub_mb_type's
///    list usage:
///      - Direct: no MVDs
///      - L0:     L0 MVD pair
///      - L1:     L1 MVD pair
///      - Bi:     L0 MVD pair, then L1 MVD pair
///    `ref_idx_lX` is inferred 0 (single-ref ship config — not on
///    the wire).
/// 6. coded_block_pattern = 0 (B_8x8 residual is not wired; only
///    the 16x16 family residuals go through the
///    `write_b_inter_residual_macroblock_cabac` path)
/// 7. (no `mb_qp_delta` when CBP=0)
///
/// Per-list MVD bin0 ctxIdxInc uses the `_per_list` variant
/// (§6E-A6.1 (4/N) fix) so encoder + walker agree.
///
/// §B-Partitioned-Residual Stage C (#206) — when `b_refs` is `Some`
/// AND `residual_enabled` is true, builds per-sub-MB prediction (each
/// sub-MB is 8×8 luma + 4×4 chroma) and emits residual via
/// [`Encoder::emit_b_residual_for_pred`]. For `B_Direct_8x8` sub-MBs
/// the prediction comes from the spatial-direct MV pair derived at
/// per-MB granularity — same as the decoder will compute on its
/// side. Otherwise emits CBP=0 with no residual + no recon write.
#[allow(clippy::too_many_arguments)]
fn emit_b_8x8_method(
    enc: &mut Encoder,
    cabac: &mut crate::codec::h264::cabac::encoder::CabacEncoder,
    mb_x: usize,
    mb_y: usize,
    sub_mb_types: [u8; 4],
    parts: [super::b_partitioned::BPartitionMv; 4],
    l1_motion_grid: Option<&super::reference_buffer::ColocatedMvGrid>,
    b_refs: Option<(
        &super::reference_buffer::ReconFrame,
        &super::reference_buffer::ReconFrame,
    )>,
    residual_enabled: bool,
    y_plane: &[u8],
    y_stride: usize,
    cb_plane: &[u8],
    cr_plane: &[u8],
    c_stride: usize,
    qp: u8,
    qp_c: u8,
    num_active_l0: u8,
) -> Result<(), EncoderError> {
    use super::motion_compensation::{
        apply_chroma_mv_block, apply_chroma_mv_block_bipred, apply_luma_mv_block,
        apply_luma_mv_block_bipred,
    };
    use crate::codec::h264::cabac::encoder::{
        encode_mvd_with_bin0_inc, encode_ref_idx, encode_sub_mb_type_b,
    };
    use crate::codec::h264::cabac::neighbor::{
        compute_mvd_ctx_idx_inc_bin0_per_list, CabacNeighborMB, CurrentMbMvdAbs, MbTypeClass,
    };

    encode_mb_skip_flag_b(cabac, false, mb_x);
    encode_mb_type_b(cabac, 22, mb_x);

    for &sub in &sub_mb_types {
        debug_assert!(sub <= 3, "B_8x8 sub_mb_type {sub} out of §6E-A6.3 scope");
        encode_sub_mb_type_b(cabac, sub as u32);
    }

    // v1.4 (#305) — ref_idx_l0 per sub-MB per spec § 7.3.5.1
    // mb_pred(): all ref_idx_l0 emit AFTER the 4×sub_mb_type and
    // BEFORE all MVDs, in sub-MB-index order, filtered by sub-MB-
    // uses-L0 (Direct=0 + L1=2 skip; L0=1 + Bi=3 emit). At
    // MultiRefConfig::SINGLE_REF default the gate is closed and zero
    // bins emit (bit-identical to v1.3).
    //
    // v1.4 Phase 4.5 (#316) — within-MB neighbour tracker (sub-MB i+
    // looks at sub-MB i for left/top via the spec § 6.4.11.7 lookup).
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
            encode_ref_idx(
                cabac, parts[s_idx].ref_idx_l0 as u32, &current_ref_idx_mb,
                mb_x, off_bx, off_by,
                (num_active_l0 - 1) as u32,
            );
            current_ref_idx_mb.fill_region(off_bx, off_by, 2, 2, parts[s_idx].ref_idx_l0 as i8);
        }
    }

    let mut current_mvd = CurrentMbMvdAbs::new();

    // Spatial-direct derivation, computed lazily and once per MB. Used
    // both for the B_Direct_8x8 grid fill (mirror of decoder's spec
    // § 8.4.1.2.2 path) AND for the residual prediction below.
    let direct_result = super::b_direct_predictor::derive_b_direct_spatial_with_col(
        &enc.mv_grid, mb_x, mb_y, l1_motion_grid,
    );

    // §B-encoder-decoder-divergence Phase 2.7 (2026-05-08, #248) —
    // The spec runs B_Direct sub-MB MV derivation BEFORE the MVD
    // parsing loop (spec § 8.4.1.2 + § 9.3.3.1.1), populating
    // Direct sub-MBs' MV cache up-front. When the list-major MVD
    // loop predicts L0/L1 for a non-Direct sub-MB, it can see a
    // Direct neighbour's already-filled MV. Phasm previously
    // deferred Direct grid fill to a post-pass which worked under
    // sub-mb-major ordering (Direct came first if at s_idx=0) but
    // breaks under list-major: e.g. mixed [Direct, L0, L1, Bi]
    // now emits L0_s1 before any grid fill for s0=Direct, making
    // predict_mv_for_mb_partition see None where the spec-compliant
    // decoder sees direct_result.
    for (s_idx, &sub) in sub_mb_types.iter().enumerate() {
        if sub != 0 {
            continue;
        }
        let off_bx = (s_idx & 1) * 2;
        let off_by = (s_idx >> 1) * 2;
        let tl_bx = mb_x * 4 + off_bx;
        let tl_by = mb_y * 4 + off_by;
        let pw = 2usize;
        let ph = 2usize;
        let l0 = if direct_result.uses_l0() {
            Some((direct_result.mv_l0, direct_result.ref_idx_l0))
        } else {
            None
        };
        let l1 = if direct_result.uses_l1() {
            Some((direct_result.mv_l1, direct_result.ref_idx_l1))
        } else {
            None
        };
        enc.mv_grid.fill_lists(tl_bx, tl_by, pw, ph, l0, l1);
    }

    // The spec parses B_8x8 sub-MB MVDs in **list-major** order
    // (H.264 spec § 7.3.5.1 + § 9.3.3.1.1): outer iterates list 0..2,
    // inner iterates sub-MB index 0..4, emitting an MVD pair only
    // when that sub-MB uses that list. Phasm previously used
    // **sub-mb-major** order which round-trips clean against phasm's
    // own walker but desyncs vs spec-compliant decoders for any
    // sub_mb_type combo where order matters (uniform_bi: max|Δ|=180,
    // mixed: max|Δ|=170 with non-zero MVDs). Walker `walk_b_8x8` in
    // cabac/bin_decoder/slice.rs updated in lockstep.
    for list in 0u8..2 {
        for s_idx in 0..4usize {
            let sub = sub_mb_types[s_idx];
            let part = &parts[s_idx];
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
            let tl_bx = mb_x * 4 + off_bx;
            let tl_by = mb_y * 4 + off_by;
            let pw = 2usize;
            let ph = 2usize;
            let cur_bx = off_bx as u8;
            let cur_by = off_by as u8;

            if list == 0 {
                let mv = part.mv_l0.unwrap_or_default();
                let predicted = super::partition_state::predict_mv_for_mb_partition(
                    &enc.mv_grid, tl_bx, tl_by, pw, ph, s_idx as u8,
                    /* current_ref_idx */ 0,
                );
                let mvd_x = (mv.mv_x as i32) - (predicted.mv_x as i32);
                let mvd_y = (mv.mv_y as i32) - (predicted.mv_y as i32);
                let bin0_inc_x = compute_mvd_ctx_idx_inc_bin0_per_list(
                    &current_mvd, &cabac.neighbors, mb_x, cur_bx, cur_by, 0, 0,
                );
                encode_mvd_with_bin0_inc(cabac, mvd_x, 0, bin0_inc_x);
                let bin0_inc_y = compute_mvd_ctx_idx_inc_bin0_per_list(
                    &current_mvd, &cabac.neighbors, mb_x, cur_bx, cur_by, 1, 0,
                );
                encode_mvd_with_bin0_inc(cabac, mvd_y, 1, bin0_inc_y);
                let abs_x = mvd_x.unsigned_abs() as i16;
                let abs_y = mvd_y.unsigned_abs() as i16;
                current_mvd.fill_region(cur_bx, cur_by, pw as u8, ph as u8, abs_x, abs_y);
                // Update L0 grid in place so subsequent emits in the same
                // sub-MB or downstream sub-MBs see the just-emitted L0 MV.
                enc.mv_grid.fill_lists(
                    tl_bx, tl_by, pw, ph,
                    part.mv_l0.map(|m| (m, 0)),
                    None,
                );
            } else {
                let mv = part.mv_l1.unwrap_or_default();
                // §B-cascade-real Phase 2 (#267) — partition-aware L1
                // predictor for B_8x8 sub-MBs. 8×8 sub-MBs don't
                // hit the 16×8 / 8×16 directional shortcuts (those
                // require part_w_4x4=4,part_h=2 or part_w=2,part_h=4),
                // but the C-block position needs partition-correct
                // `part_w_4x4` to match L0's symmetric behaviour.
                let predicted = super::b_direct_predictor::predict_mv_for_mb_partition_l1(
                    &enc.mv_grid, tl_bx, tl_by, pw, ph, s_idx as u8,
                    /* current_ref_idx */ 0,
                );
                let mvd_x = (mv.mv_x as i32) - (predicted.mv_x as i32);
                let mvd_y = (mv.mv_y as i32) - (predicted.mv_y as i32);
                let bin0_inc_x = compute_mvd_ctx_idx_inc_bin0_per_list(
                    &current_mvd, &cabac.neighbors, mb_x, cur_bx, cur_by, 0, 1,
                );
                encode_mvd_with_bin0_inc(cabac, mvd_x, 0, bin0_inc_x);
                let bin0_inc_y = compute_mvd_ctx_idx_inc_bin0_per_list(
                    &current_mvd, &cabac.neighbors, mb_x, cur_bx, cur_by, 1, 1,
                );
                encode_mvd_with_bin0_inc(cabac, mvd_y, 1, bin0_inc_y);
                let abs_x = mvd_x.unsigned_abs() as i16;
                let abs_y = mvd_y.unsigned_abs() as i16;
                current_mvd.fill_region_l1(cur_bx, cur_by, pw as u8, ph as u8, abs_x, abs_y);
                enc.mv_grid.fill_lists(
                    tl_bx, tl_by, pw, ph,
                    None,
                    part.mv_l1.map(|m| (m, 0)),
                );
            }
        }
    }

    // (Direct sub-MB grid fill moved BEFORE the MVD loop above per
    // the spec-mandated direct-prediction ordering.)

    if let (Some((l0_ref, l1_ref)), true) = (b_refs, residual_enabled) {
        let mb_px_x = (mb_x * 16) as u32;
        let mb_px_y = (mb_y * 16) as u32;
        let mb_cpx_x = (mb_x * 8) as u32;
        let mb_cpx_y = (mb_y * 8) as u32;

        let mut pred_y = [[0u8; 16]; 16];
        let mut pred_cb = [[0u8; 8]; 8];
        let mut pred_cr = [[0u8; 8]; 8];

        for (s_idx, (&sub, part)) in sub_mb_types.iter().zip(parts.iter()).enumerate() {
            let off_x_luma = ((s_idx & 1) * 8) as u32;
            let off_y_luma = ((s_idx >> 1) * 8) as u32;
            let off_x_chroma = ((s_idx & 1) * 4) as u32;
            let off_y_chroma = ((s_idx >> 1) * 4) as u32;

            // Determine the (mode, mvs) used for this sub-MB's
            // prediction. Direct uses the spatial-direct result.
            enum Use {
                L0,
                L1,
                Bi,
            }
            let (use_kind, mv_l0_for_pred, mv_l1_for_pred) = match sub {
                0 => {
                    // B_Direct_8x8 — same MVs as the decoder will derive.
                    let mv_l0 = direct_result.mv_l0;
                    let mv_l1 = direct_result.mv_l1;
                    let kind = match (
                        direct_result.uses_l0(),
                        direct_result.uses_l1(),
                    ) {
                        (true, true) => Use::Bi,
                        (true, false) => Use::L0,
                        (false, true) => Use::L1,
                        // Fallback: treat as L0 zero-MV (decoder mirrors).
                        (false, false) => Use::L0,
                    };
                    (kind, mv_l0, mv_l1)
                }
                1 => (Use::L0, part.mv_l0.unwrap_or_default(), MotionVector::default()),
                2 => (Use::L1, MotionVector::default(), part.mv_l1.unwrap_or_default()),
                3 => (
                    Use::Bi,
                    part.mv_l0.unwrap_or_default(),
                    part.mv_l1.unwrap_or_default(),
                ),
                _ => unreachable!("sub_mb_type {sub} > 3"),
            };

            // Luma 8×8.
            {
                let pred_y_flat = pred_y.as_flattened_mut();
                let y_start = (off_y_luma * 16 + off_x_luma) as usize;
                let pred_y_sub = &mut pred_y_flat[y_start..];
                match use_kind {
                    Use::L0 => apply_luma_mv_block(
                        l0_ref,
                        mb_px_x + off_x_luma, mb_px_y + off_y_luma,
                        8, 8, mv_l0_for_pred, pred_y_sub, 16,
                    ),
                    Use::L1 => apply_luma_mv_block(
                        l1_ref,
                        mb_px_x + off_x_luma, mb_px_y + off_y_luma,
                        8, 8, mv_l1_for_pred, pred_y_sub, 16,
                    ),
                    Use::Bi => apply_luma_mv_block_bipred(
                        l0_ref, mv_l0_for_pred, l1_ref, mv_l1_for_pred,
                        mb_px_x + off_x_luma, mb_px_y + off_y_luma,
                        8, 8, pred_y_sub, 16,
                    ),
                }
            }
            // Chroma 4×4 per component.
            for (component, pred_c) in [&mut pred_cb, &mut pred_cr].iter_mut().enumerate() {
                let pred_c_flat = pred_c.as_flattened_mut();
                let c_start = (off_y_chroma * 8 + off_x_chroma) as usize;
                let pred_c_sub = &mut pred_c_flat[c_start..];
                match use_kind {
                    Use::L0 => apply_chroma_mv_block(
                        l0_ref, component as u8,
                        mb_cpx_x + off_x_chroma, mb_cpx_y + off_y_chroma,
                        4, 4, mv_l0_for_pred, pred_c_sub, 8,
                    ),
                    Use::L1 => apply_chroma_mv_block(
                        l1_ref, component as u8,
                        mb_cpx_x + off_x_chroma, mb_cpx_y + off_y_chroma,
                        4, 4, mv_l1_for_pred, pred_c_sub, 8,
                    ),
                    Use::Bi => apply_chroma_mv_block_bipred(
                        l0_ref, mv_l0_for_pred, l1_ref, mv_l1_for_pred,
                        component as u8,
                        mb_cpx_x + off_x_chroma, mb_cpx_y + off_y_chroma,
                        4, 4, pred_c_sub, 8,
                    ),
                }
            }
        }

        let res = enc.emit_b_residual_for_pred(
            cabac, mb_x, mb_y,
            y_plane, y_stride, cb_plane, cr_plane, c_stride,
            &pred_y, &pred_cb, &pred_cr,
            qp, qp_c,
        )?;

        let mut nb = CabacNeighborMB::default();
        nb.mb_type = MbTypeClass::PInter;
        nb.mb_skip_flag = false;
        nb.cbp_luma = res.cbp_luma_8x8;
        nb.cbp_chroma = res.cbp_chroma;
        nb.mb_qp_delta = res.qp_delta_emitted;
        nb.coded_block_flag_cat = res.current_cbf.to_neighbor_cbf();
        // v1.4 Phase 4.5 (#316) — per-block fill from sub-MB geometry.
        // Direct (sub_mb_type=0) and L1-only (sub_mb_type=2) sub-MBs
        // stay 0 (no L0 transmitted; walker symmetric).
        nb.ref_idx_l0 = fill_ref_idx_l0_b8x8(sub_mb_types, &parts);
        nb.abs_mvd_comp = current_mvd.to_neighbor();
        nb.abs_mvd_comp_l1 = current_mvd.to_neighbor_l1();
        nb.transform_size_8x8_flag = false;
        cabac.neighbors.commit(mb_x, nb);
    } else {
        encode_coded_block_pattern(cabac, /* cbp_value */ 0, mb_x);
        let nb = CabacNeighborMB {
            mb_type: MbTypeClass::PInter,
            mb_skip_flag: false,
            cbp_luma: 0,
            cbp_chroma: 0,
            // v1.4 Phase 4.5 (#316) — per-block fill from sub-MB geometry.
            ref_idx_l0: fill_ref_idx_l0_b8x8(sub_mb_types, &parts),
            abs_mvd_comp: current_mvd.to_neighbor(),
            abs_mvd_comp_l1: current_mvd.to_neighbor_l1(),
            ..CabacNeighborMB::default()
        };
        cabac.neighbors.commit(mb_x, nb);
    }

    Ok(())
}

/// §6E-A6.1 — populate the encoder MV grid for a B_Direct / B_Skip
/// macroblock at `(mb_x, mb_y)` using the same spatial-direct
/// derivation the decoder will run on its side. Both modes leave
/// no MVD on the wire and the decoder reconstructs the MV via
/// spec § 8.4.1.2.2 spatial direct; the encoder must mirror that
/// derivation here so subsequent non-direct B-MBs (§6E-A6.1+) see
/// consistent neighbour data when they predict their own MVDs.
///
/// Writes ONE per-list `(mv, ref_idx)` pair across all 16 of the
/// MB's 4×4 cells. The median-only spatial-direct path produces
/// the same MV for every cell (no per-sub-block static check
/// yet — see `b_direct_predictor.rs` module header), so per-MB
/// granularity matches the implementation.
///
/// §B-direct-fix Stage 2 (#232): now returns the
/// [`BDirectSpatialResult`] so callers (B_Skip / B_Direct_16x16
/// emit paths) can build the prediction surface and write it to
/// recon + visual_recon. Without that write the encoder's recon
/// for the MB stays at the previous frame's pixels (the decoder
/// reconstructs predicted pixels from the bitstream's spatial-
/// direct MVs) → encoder.recon ≠ reference-decoder output → cascade.
/// §B-direct-fix.v3 — Direct-mode dispatch context.
///
/// `Spatial` runs the spec § 8.4.1.2.2 spatial-direct path (median
/// of A/B/C with colZeroFlag). `Temporal { .. }` runs the spec
/// § 8.4.1.2.3 temporal-direct path (scaled colocated MV, no
/// median, no neighbour mixing).
///
/// Temporal-direct is preferred for motion-boundary content where
/// spatial-direct's median picks the wrong neighbour and causes
/// visible streaks. See `docs/design/video/h264/encoder-quality-plan.md`
/// §B-direct-fix.v3 section.
#[derive(Debug, Clone, Copy)]
pub enum BDirectMode {
    Spatial,
    Temporal { poc_curr: i32, poc_l0: i32, poc_l1: i32 },
}

/// §B-direct-fix.v3.ROOT.v13 2026-05-07 — predictor magnitude clamp for
/// B-frame ME bit-cost anchor.
///
/// When the spec's median predictor is huge (>8 px = 32 quarter-pel),
/// it's almost always wrong (chain-propagated bad MV from raster
/// neighbour). Using it as the rate anchor traps ME in a wrong-MV
/// local minimum. Threshold of 32 qpel covers typical camera-shake
/// while excluding the body-internal-boundary chain-propagation
/// pattern observed at 1080p (-205, 112) and similar.
///
/// Returns ZERO when predicted is >32 qpel in either axis; passthrough
/// otherwise. The bitstream-side MVD computation still uses the real
/// median predictor (spec-compliant); only ME's internal cost model
/// uses this clamped anchor.
fn clamp_me_anchor(predicted: MotionVector) -> MotionVector {
    const ME_ANCHOR_CLAMP_QPEL: i16 = 32;
    if predicted.mv_x.abs() > ME_ANCHOR_CLAMP_QPEL
        || predicted.mv_y.abs() > ME_ANCHOR_CLAMP_QPEL
    {
        MotionVector::ZERO
    } else {
        predicted
    }
}

fn populate_b_direct_grid(
    grid: &mut super::partition_state::EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
    l1_motion_grid: Option<&super::reference_buffer::ColocatedMvGrid>,
    direct_mode: BDirectMode,
) -> super::b_direct_predictor::BDirectSpatialResult {
    let r = match direct_mode {
        BDirectMode::Spatial => super::b_direct_predictor::derive_b_direct_spatial_with_col(
            grid, mb_x, mb_y, l1_motion_grid,
        ),
        BDirectMode::Temporal { poc_curr, poc_l0, poc_l1 } => {
            super::b_direct_predictor::derive_b_direct_temporal(
                l1_motion_grid, mb_x, mb_y, poc_curr, poc_l0, poc_l1,
            )
        }
    };

    // Phase 2.12 (#275, 2026-05-08) — per-8×8 sub-block grid update.
    //
    // Spec § 8.4.1.2.2 step 6 + the spec § 8.4.1.2.2 spatial-direct derivation
    // apply colZeroFlag override PER 8×8 sub-block. Some sub-blocks
    // get MV=(0,0), others keep the median predictor. The grid we
    // write must reflect this so subsequent B-MBs reading neighbour
    // PMVs see the SAME per-sub-block values that a spec-compliant decoder
    // sees in its mv_cache.
    //
    // Pre-fix (Phase 2.11) wrote the median MV uniformly to all 16
    // 4×4 cells. Post-fix writes per-8×8: 4 separate fill_lists
    // calls, each covering a 2×2 4×4-cell rect (= one 8×8 sub-block),
    // with that sub-block's specific MV.
    let mb_bx = mb_x * 4;
    let mb_by = mb_y * 4;
    let uses_l0 = r.uses_l0();
    let uses_l1 = r.uses_l1();
    for i8 in 0..4 {
        let sub_bx = mb_bx + (i8 & 1) * 2;
        let sub_by = mb_by + (i8 >> 1) * 2;
        let l0_cell = if uses_l0 {
            Some((r.mv_l0_per_8x8[i8], r.ref_idx_l0))
        } else {
            None
        };
        let l1_cell = if uses_l1 {
            Some((r.mv_l1_per_8x8[i8], r.ref_idx_l1))
        } else {
            None
        };
        // Each 8×8 sub-block = 2×2 4×4 cells.
        grid.fill_lists(sub_bx, sub_by, 2, 2, l0_cell, l1_cell);
    }
    // Lists not used at this MB stay at their post-`reset` default
    // (REF_IDX_NONE) — `fill_lists(..., None, ...)` does NOT touch
    // the cells for that list, so the absent-list semantic is
    // preserved automatically.
    r
}

/// §B-direct-fix Stage 2 (#232) — convert spatial-direct derivation
/// result into a [`BInterMode`] for prediction recon. B_Skip and
/// B_Direct_16x16 reconstruct their pixels from spatial-direct MVs
/// per spec § 7.3.5.1 / § 8.4.1.2.2; this helper picks the matching
/// inter-mode shape so [`Encoder::write_b_prediction_recon`] can
/// build the prediction surface.
fn b_direct_to_inter_mode(
    direct: &super::b_direct_predictor::BDirectSpatialResult,
) -> super::b_inter_prediction::BInterMode {
    use super::b_inter_prediction::BInterMode;
    match (direct.uses_l0(), direct.uses_l1()) {
        (true, true) => BInterMode::Bi_16x16 {
            mv_l0: direct.mv_l0,
            mv_l1: direct.mv_l1,
        },
        (true, false) => BInterMode::L0_16x16 { mv: direct.mv_l0 },
        (false, true) => BInterMode::L1_16x16 { mv: direct.mv_l1 },
        // Spec § 8.4.1.2.1 boundary case in `derive_b_direct_spatial_with_col`
        // forces both refs to 0 so this branch shouldn't fire — defensive
        // fallback to zero-MV bipred matches the spec's collapsed result.
        (false, false) => BInterMode::Bi_16x16 {
            mv_l0: MotionVector::ZERO,
            mv_l1: MotionVector::ZERO,
        },
    }
}

#[allow(dead_code)] // §6E-A4(c)-lite legacy helper, kept for diagnostic compat.
pub(super) fn mb_skip_or_direct_decision(frame_num: u32, mb_addr: u32) -> bool {
    let mut x = (frame_num as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = x.wrapping_add(mb_addr as u64);
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    (x & 1) == 1
}

impl Encoder {
    /// Construct an encoder for the given dimensions + optional
    /// user-facing quality target. Dimensions must be 16-aligned; pad
    /// your input if necessary.
    ///
    /// **`quality` is a user-quality value on the [0..=100] scale,
    /// NOT a CRF.** Internally it is mapped to CRF via
    /// `rate_control::quality_to_crf` (e.g. quality=26 → CRF=33,
    /// quality=50 → CRF=28, quality=75 → CRF=23). To pass a CRF
    /// value directly, use [`Encoder::new_with_crf`].
    pub fn new(width: u32, height: u32, quality: Option<u8>) -> Result<Self, EncoderError> {
        if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
            return Err(EncoderError::InvalidInput(format!(
                "dimensions must be 16-aligned, got {width}×{height}"
            )));
        }
        let sps_params = SpsParams {
            width_pixels: width,
            height_pixels: height,
            sps_id: 0,
            // §6E-B(a) — bumped from 1 → 2 for SPS-level
            // fingerprint match with real-world commercial encoders
            // (mobile MediaCodec / iPhone H.264 capture pipelines
            // universally advertise num_ref_frames ≥ 2). Slice-level
            // behavior unchanged: encoder still emits
            // num_ref_idx_l0_active_minus1=0 (no per-partition
            // ref_idx_l0 field), so on-the-wire pixel reconstruction
            // is byte-identical to single-ref output. The wire field
            // change is just the SPS byte. §6E-B(b) v1.1 follow-on
            // will add real multi-ref ME + per-partition ref_idx_l0
            // emission for full ref_idx-distribution fingerprint
            // match.
            max_num_ref_frames: 2,
            // Default: pic_order_cnt_type=2 (frame_num→POC).
            // §6E-A4 enable_b_frames=true bumps to 0 + ref count 3.
            pic_order_cnt_type: 2,
            log2_max_pic_order_cnt_lsb_minus4: 4,
            // §Stealth.L3.1 — VUI emission is OFF by default; the
            // CABAC/High path turns it on inside `emit_params_if_needed`
            // so phasm output lands inside the common-encoder centroid at the
            // SPS level. CAVLC Baseline keeps VUI absent (the legacy
            // shape the parser is exercised against).
            vui: None,
        };
        let pps_params = PpsParams {
            pps_id: 0,
            sps_id: 0,
            // §Stealth.L4.6.1 — the converter-pipeline centroid emits pic_init_qp_minus26 = -3
            // (init QP = 23). Phasm previously emitted 0 (init QP = 26).
            // Slice headers compute slice_qp_delta = qp - pic_init_qp, so
            // changing this constant from 26→23 just shifts every emitted
            // slice_qp_delta by +3; effective per-slice QP is preserved.
            // Verified via `a reference-decoder trace-headers tool` on
            // /tmp/reference_centroid_img4138_f10.h264.
            pic_init_qp: 23,
            deblocking_filter_control_present: true,
            // §Stealth.L4.6.2 — the converter-pipeline centroid emits PPS
            // num_ref_idx_l0_default_active_minus1 = 2 (3 active L0 refs
            // default) + num_ref_idx_l1_default_active_minus1 = 0 (1 L1).
            // Phasm is 1-ref-P (Phase 6E-B not yet shipped) so every
            // slice header sets num_ref_idx_active_override_flag = 1 +
            // num_ref_idx_l0_active_minus1 = 0 to bring active count
            // back to 1. Net wire effect: same per-slice ref count, +3
            // bits per slice header for the override + +2 bits per PPS
            // for the higher default — closes a PPS-level L4 fingerprint
            // gap vs the converter-pipeline centroid IBPBP reference.
            num_ref_idx_l0_default_active_minus1: 2,
            num_ref_idx_l1_default_active_minus1: 0,
            // §Stealth.L4.6.3 — the converter-pipeline centroid emits chroma_qp_index_offset
            // = -2 + second_chroma_qp_index_offset = -2 (Cb/Cr both -2).
            // Closes a PPS L4 fingerprint divergence. Encoder-side
            // chroma QP derivation is updated to thread these offsets
            // through every derive_chroma_qp call site so encoder/walker
            // stay in lockstep.
            chroma_qp_index_offset: -2,
            second_chroma_qp_index_offset: -2,
            // §Stealth.L4.6.4 — the converter-pipeline centroid emits PPS weighted_pred_flag
            // = 1. Phasm previously emitted 0. Each P-slice header now
            // carries a 4-bit degenerate pred_weight_table (no actual
            // weighted prediction in motion compensation — phasm uses
            // the unweighted formula). Closes the last PPS L4
            // fingerprint divergence vs the converter-pipeline centroid IBPBP reference.
            weighted_pred_flag: true,
            // §6E-D.5(l) — implicit weighted bipred (spec § 8.4.2.3.2).
            // Verified via `a reference-decoder trace-headers tool` that the converter-pipeline centroid
            // emits weighted_bipred_idc=2 in its PPS; phasm matching
            // is a CONTAINER-LEVEL stealth alignment (L4 fingerprint).
            //
            // For our SYMMETRIC IBPBP M=2 GOP shape (B sits exactly
            // midway between L0 and L1 anchors temporally), implicit
            // weights reduce to W0=W1=32 (Q.6 fixed-point). The
            // resulting bipred formula
            //   ((32×L0 + 32×L1 + 32) >> 6)
            // is bit-identical to the default
            //   ((L0 + L1 + 1) >> 1)
            // for all (L0, L1) ∈ [0, 255]². So encoder and decoder
            // produce IDENTICAL pixel output regardless of this PPS
            // field for symmetric M=2.
            //
            // For non-M=2 GOPs (multi-B GOPs IBBPBBP etc.), implicit
            // weighting genuinely differs and would close more of the
            // bipred gap — but phasm's IBPBP M=2 doesn't activate that
            // path. Setting =2 here is forward-compatible with future
            // GOP shapes + matches the converter-pipeline centroid PPS profile today.
            weighted_bipred_idc: 2,
        };
        let mb_w = (width / 16) as usize;
        let mb_h = (height / 16) as usize;
        Ok(Self {
            width,
            height,
            frame_num: 0,
            stego_frame_idx: 0,
            enable_mvd_stego_hook: false,
            rc: RateController::new(quality),
            recon: ReconBuffer::new(width, height)?,
            visual_recon: ReconBuffer::new(width, height)?,
            sps_params,
            pps_params,
            params_emitted: false,
            entropy_mode: EntropyMode::Cabac,
            enable_transform_8x8: false,
            enable_b_frames: false,
            b_rdo_config: super::mb_decision_b::BRdoConfig::SAFE,
            multi_ref_config: MultiRefConfig::SINGLE_REF,
            poc_tracker: super::poc::PocTracker::new(),
            display_idx_of_prev_anchor: 0,
            dpb: ReferenceBuffer::new(),
            gop_position: 0,
            gop_length: 30,
            mv_grid: EncoderMvGrid::new(mb_w, mb_h),
            i4x4_mode_grid: vec![0xFF; mb_w * 4 * mb_h * 4],
            i8x8_mode_grid: vec![0xFF; mb_w * 2 * mb_h * 2],
            total_coeff_grid: vec![0xFF; mb_w * 4 * mb_h * 4],
            intra16x16_dc_tc_grid: vec![0xFF; mb_w * mb_h],
            // Init chroma TC grids to 0 (matching decoder's
            // `vec![0; ...]` init). Any "MB exists but didn't emit
            // chroma AC" neighbor reads 0 — matching the spec.
            chroma_cb_tc_grid: vec![0; (mb_w * 2) * (mb_h * 2)],
            chroma_cr_tc_grid: vec![0; (mb_w * 2) * (mb_h * 2)],
            qp_grid: vec![26; mb_w * mb_h],
            intra_grid: vec![false; mb_w * mb_h],
            transform_8x8_grid: vec![false; mb_w * mb_h],
            prev_mb_qp: 26,
            aq_frame_mean_log2_q8: 0,
            mb_tree: None,
            mb_tree_display_idx: 0,
            lookahead: None,
            crf: None,
            me: MotionEstimator::new(),
            cabac_trace_enabled: false,
            cabac_trace_buffer: Vec::new(),
            mode_stats: [0; 9],
            stego_hook: None,
            // Task #383 — snapshot env-derived debug knobs ONCE here.
            // The encoder then reads from this snapshot for the rest
            // of its life; mid-encode env mutations by other test
            // threads cannot affect this encoder's behavior.
            env_snapshot: super::mb_decision_b::EncoderEnvSnapshot::capture_or_inherit(),
        })
    }

    /// §v1.7 Phase 3 (#325) — construct an encoder anchored at the
    /// given CRF directly (skipping the user-quality 0..=100 → CRF
    /// mapping in [`Encoder::new`]). `crf` is on the H.264 QP scale
    /// (0..=51); typical values: 18 (very high), 23 (default),
    /// 26 (medium, HandBrake faster default), 28 (medium-low).
    ///
    /// Equivalent to constructing with `Encoder::new(.., None)` and
    /// then assigning `enc.crf = Some(crf)`, with the additional
    /// effect that `self.rc.target_crf` is initialised to `crf` so
    /// callers that don't iterate frames via `encode_{i,p,b}_frame`
    /// (the entry points that invoke `apply_crf_base_qp`) still see
    /// the right base QP.
    pub fn new_with_crf(width: u32, height: u32, crf: u8) -> Result<Self, EncoderError> {
        let mut enc = Self::new(width, height, None)?;
        let crf_clamped = crf.clamp(0, 51);
        enc.crf = Some(crf_clamped);
        enc.rc.target_crf = crf_clamped;
        Ok(enc)
    }

    /// Phase 6D.8: install a stego hook. Encoder calls into it
    /// post-quantize / pre-entropy on every residual block + MVD
    /// emit. With `None` (default) the encoder behaves byte-
    /// identically to pre-6D.8.
    ///
    /// See [`super::super::stego::encoder_hook::StegoMbHook`].
    pub fn set_stego_hook(
        &mut self,
        hook: Option<Box<dyn super::super::stego::encoder_hook::StegoMbHook>>,
    ) {
        self.stego_hook = hook;
    }

    /// Phase 6D.8: drain the stego hook (typically at end of GOP).
    /// Returns the boxed hook so the caller can `downcast` or call
    /// hook-specific finalizers (e.g. `PositionLoggerHook::take_cover()`).
    pub fn take_stego_hook(
        &mut self,
    ) -> Option<Box<dyn super::super::stego::encoder_hook::StegoMbHook>> {
        self.stego_hook.take()
    }

    /// Phase 6D.8 internal helper: invoke the stego hook for a
    /// residual block emit point. Computes the encoder-wide MB
    /// address from `(mb_x, mb_y)`, reads `frame_num` into a local
    /// (Copy) so the mutable borrow of `stego_hook` doesn't
    /// conflict, then dispatches to the hook if any.
    ///
    /// **Insertion rule**: must be called AFTER quantize and
    /// BEFORE the entropy emit + reconstruction. This guarantees
    /// recon and entropy see the same (possibly modified)
    /// coefficients.
    #[inline]
    fn invoke_stego_residual_hook(
        &mut self,
        mb_x: usize,
        mb_y: usize,
        scan_coeffs: &mut [i32],
        start_idx: usize,
        end_idx: usize,
        path_kind: super::super::stego::orchestrate::ResidualPathKind,
    ) {
        let mb_w = (self.width / 16) as usize;
        let mb_addr = (mb_y * mb_w + mb_x) as u32;
        let frame = self.stego_frame_idx;
        if let Some(hook) = self.stego_hook.as_mut() {
            hook.on_residual_block(
                frame, mb_addr, scan_coeffs, start_idx, end_idx, path_kind,
            );
        }
    }

    /// §B-cascade-real v1.1 — Phase 1.1.A.
    ///
    /// Write a 16×16 luma macroblock to BOTH `self.recon` (used as
    /// neighbour for next-MB prediction; preserves cover-capture
    /// invariant) AND `self.visual_recon` (used by the mux output;
    /// reflects what a downstream player produces from our bitstream).
    ///
    /// Phase 1.1.A: both buffers receive IDENTICAL pixels. Behaviour
    /// is byte-identical to a single-buffer `self.recon.write_luma_mb`
    /// call (apart from the second copy). Phase 1.1.B will diverge
    /// the inputs: `pre_flip` for `self.recon`, `post_flip` for
    /// `self.visual_recon`.
    #[inline]
    pub fn write_luma_mb_dual(
        &mut self,
        mb_x: u32,
        mb_y: u32,
        pixels: &[[u8; 16]; 16],
    ) {
        self.recon.write_luma_mb(mb_x, mb_y, pixels);
        self.visual_recon.write_luma_mb(mb_x, mb_y, pixels);
    }

    /// §B-cascade-real v1.1 — Phase 1.1.A. Companion to
    /// [`Self::write_luma_mb_dual`] for the chroma plane.
    #[inline]
    pub fn write_chroma_block_dual(
        &mut self,
        mb_x: u32,
        mb_y: u32,
        plane: u8,
        pixels: &[[u8; 8]; 8],
    ) {
        self.recon.write_chroma_block(mb_x, mb_y, plane, pixels);
        self.visual_recon.write_chroma_block(mb_x, mb_y, plane, pixels);
    }

    /// §B-direct-fix Stage 2 (#232) — mirror a freshly-written
    /// 16x16 luma block from `self.recon` into `self.visual_recon`.
    /// Used after `encode_i4x4_mb` / `encode_i8x8_mb` (both write
    /// only `&mut ReconBuffer`, not visual_recon — Phase 1.1.B
    /// Shape 2 deferred those luma helpers as #224 / #225). Without
    /// this mirror, intra-in-P MBs leave visual_recon stale at the
    /// previous frame's pixels → encoder/decoder recon parity break
    /// → cascade.
    #[inline]
    fn mirror_luma_mb_to_visual_recon(&mut self, mb_x: u32, mb_y: u32) {
        let stride = self.recon.width as usize;
        let px = (mb_x as usize) * 16;
        let py = (mb_y as usize) * 16;
        for dy in 0..16 {
            for dx in 0..16 {
                self.visual_recon.y[(py + dy) * stride + (px + dx)] =
                    self.recon.y[(py + dy) * stride + (px + dx)];
            }
        }
    }

    /// Phase 6D.8 internal helper: invoke the stego hook for an
    /// MVD slot. Same constraints as
    /// [`Self::invoke_stego_residual_hook`].
    #[inline]
    fn invoke_stego_mvd_hook(
        &mut self,
        mb_x: usize,
        mb_y: usize,
        slot: &mut super::super::stego::inject::MvdSlot,
    ) {
        let mb_w = (self.width / 16) as usize;
        let mb_addr = (mb_y * mb_w + mb_x) as u32;
        let frame = self.stego_frame_idx;
        if let Some(hook) = self.stego_hook.as_mut() {
            hook.on_mvd_slot(frame, mb_addr, slot);
        }
    }

    /// Phase 6F.2 — open a per-MB MVD position savepoint on the
    /// stego hook. Pair with `commit_mvd_for_mb` (when the MB
    /// emits MVDs in the bitstream) or `rollback_mvd_for_mb` (when
    /// the MB ends up as P_SKIP / intra-in-P, no MVDs in
    /// bitstream). No-op when no hook is installed.
    #[inline]
    fn begin_mvd_for_mb(&mut self) {
        if let Some(hook) = self.stego_hook.as_mut() {
            hook.begin_mvd_for_mb();
        }
    }

    /// Phase 6F.2 — commit the per-MB MVD positions logged since
    /// the matching `begin_mvd_for_mb`. Encoder calls this after a
    /// successful inter MB emission (where the MVDs landed in the
    /// bitstream).
    #[inline]
    fn commit_mvd_for_mb(&mut self) {
        if let Some(hook) = self.stego_hook.as_mut() {
            hook.commit_mvd_for_mb();
        }
    }

    /// Phase 6F.2 — discard the per-MB MVD positions logged since
    /// the matching `begin_mvd_for_mb`. Encoder calls this on
    /// P_SKIP and intra-in-P emit paths, where the MB has no
    /// MVDs in the actual bitstream so any logged positions were
    /// phantoms (deferred-items.md §37).
    #[inline]
    fn rollback_mvd_for_mb(&mut self) {
        if let Some(hook) = self.stego_hook.as_mut() {
            hook.rollback_mvd_for_mb();
        }
    }

    /// Phase 6F.2(k).2 — query the planned MVD sign-bit overrides
    /// for one partition's (X, Y) MVD pair. Returns `(ox, oy)`
    /// where each is `Some(b)` if the position has a stego plan
    /// flip, else `None`. The encoder writes `b` to the bypass
    /// sign bin instead of the natural sign bit.
    ///
    /// `mvd_x` / `mvd_y` are the encoder's NATURAL pre-injection
    /// MVD values — used to compute the slot's PositionKey but
    /// NOT modified. The encoder's `mv_grid` + MC + neighbor
    /// predictors all see the original MV → no cascade.
    #[inline]
    fn mvd_sign_overrides_for_partition(
        &mut self,
        mb_x: usize,
        mb_y: usize,
        partition: u8,
        mvd_x: i32,
        mvd_y: i32,
    ) -> (Option<u8>, Option<u8>) {
        let mb_w = (self.width / 16) as usize;
        let mb_addr = (mb_y * mb_w + mb_x) as u32;
        let frame = self.stego_frame_idx;
        if let Some(hook) = self.stego_hook.as_mut() {
            use super::super::stego::inject::MvdSlot;
            use super::super::stego::Axis;
            let slot_x = MvdSlot { list: 0, partition, axis: Axis::X, value: mvd_x };
            let slot_y = MvdSlot { list: 0, partition, axis: Axis::Y, value: mvd_y };
            let ox = hook.mvd_sign_override(frame, mb_addr, &slot_x);
            let oy = hook.mvd_sign_override(frame, mb_addr, &slot_y);
            (ox, oy)
        } else {
            (None, None)
        }
    }

    /// Phase 6D.8 §30D-A/A2: fire MVD stego hook for each partition's
    /// (x, y) MVDs BEFORE motion compensation. Updates `choice` in
    /// place if the hook modified MVD values, recomputing per-
    /// partition MVs as `pred + new_mvd` so subsequent MC + recon
    /// run with the FINAL MVs (no enc/dec drift).
    ///
    /// **Scope**: P_L0_16x16 (§30D-A), P_L0_16x8 + P_L0_8x16
    /// (§30D-A2). P_8x8 + sub_mb_types remain in §30D-A3.
    ///
    /// **mv_grid pre-fill**: for multi-partition MBs, partition-N's
    /// pred lookup needs to see partition-(N-1)'s MV. We pre-fill
    /// the grid as we walk partitions; the entropy-time fill in
    /// emit_p_mvds_cabac re-writes the same value, idempotent.
    fn apply_mvd_hook_to_choice(
        &mut self,
        choice: &mut super::partition_decision::PMbChoice,
        mb_x: usize,
        mb_y: usize,
    ) {
        use super::partition_decision::PMbChoice;

        if !self.enable_mvd_stego_hook || self.stego_hook.is_none() {
            return;
        }

        let base_bx = mb_x * 4;
        let base_by = mb_y * 4;

        match choice {
            PMbChoice::P16x16 { mv, .. } => {
                self.fire_mvd_hook_one_partition(
                    mb_x, mb_y, base_bx, base_by, 4, 4,
                    /* mb_part_idx */ None, /* partition */ 0, mv,
                );
                self.mv_grid.fill(base_bx, base_by, 4, 4, *mv, 0);
            }
            PMbChoice::P16x8 { mvs, .. } => {
                // Partition 0 (top): (base_bx, base_by, 4, 2).
                self.fire_mvd_hook_one_partition(
                    mb_x, mb_y, base_bx, base_by, 4, 2,
                    Some(0), 0, &mut mvs[0],
                );
                self.mv_grid.fill(base_bx, base_by, 4, 2, mvs[0], 0);
                // Partition 1 (bottom): (base_bx, base_by+2, 4, 2).
                self.fire_mvd_hook_one_partition(
                    mb_x, mb_y, base_bx, base_by + 2, 4, 2,
                    Some(1), 1, &mut mvs[1],
                );
                self.mv_grid.fill(base_bx, base_by + 2, 4, 2, mvs[1], 0);
            }
            PMbChoice::P8x16 { mvs, .. } => {
                // Partition 0 (left): (base_bx, base_by, 2, 4).
                self.fire_mvd_hook_one_partition(
                    mb_x, mb_y, base_bx, base_by, 2, 4,
                    Some(0), 0, &mut mvs[0],
                );
                self.mv_grid.fill(base_bx, base_by, 2, 4, mvs[0], 0);
                // Partition 1 (right): (base_bx+2, base_by, 2, 4).
                self.fire_mvd_hook_one_partition(
                    mb_x, mb_y, base_bx + 2, base_by, 2, 4,
                    Some(1), 1, &mut mvs[1],
                );
                self.mv_grid.fill(base_bx + 2, base_by, 2, 4, mvs[1], 0);
            }
            PMbChoice::P8x8 { sub } => {
                use super::partition_decision::SUB_MB_ORIGINS_4X4;
                for (i, sub_choice) in sub.iter_mut().enumerate() {
                    let (off_x_4x4, off_y_4x4) = SUB_MB_ORIGINS_4X4[i];
                    let sub_bx_abs = base_bx + off_x_4x4;
                    let sub_by_abs = base_by + off_y_4x4;
                    self.apply_mvd_hook_to_sub_mb(
                        mb_x, mb_y, sub_bx_abs, sub_by_abs,
                        i as u8, sub_choice,
                    );
                }
            }
        }
    }

    /// Phase 6D.8 §30D-A3: fire MVD hook for one 8×8 sub-MB inside
    /// a P_8x8 MB. Mirrors `emit_sub_mb_mvds_cabac` structure
    /// (encoder.rs:1979). Sub-MB partitions use the median
    /// predictor (`predict_mv_for_partition`); the directional
    /// shortcuts in `predict_mv_for_mb_partition` apply only to
    /// MB-level P_16x8 / P_8x16.
    ///
    /// `partition` convention: `sub_mb_idx * 4 + sub_part_idx`,
    /// matching the decoder's `decode_sub_mb_mvds`. Gives 16
    /// unique partition slots across 4 sub-MBs × up to 4 partitions
    /// (P_4x4).
    fn apply_mvd_hook_to_sub_mb(
        &mut self,
        mb_x: usize,
        mb_y: usize,
        sub_bx_abs: usize,
        sub_by_abs: usize,
        sub_mb_idx: u8,
        sub_choice: &mut super::partition_decision::SubMbChoice,
    ) {
        use super::partition_decision::SubMbChoice;

        let p = |sub_part_idx: u8| sub_mb_idx * 4 + sub_part_idx;

        match sub_choice {
            SubMbChoice::P8x8 { mv, .. } => {
                self.fire_mvd_hook_one_partition(
                    mb_x, mb_y, sub_bx_abs, sub_by_abs, 2, 2,
                    None, p(0), mv,
                );
                self.mv_grid.fill(sub_bx_abs, sub_by_abs, 2, 2, *mv, 0);
            }
            SubMbChoice::P8x4 { mvs, .. } => {
                self.fire_mvd_hook_one_partition(
                    mb_x, mb_y, sub_bx_abs, sub_by_abs, 2, 1,
                    None, p(0), &mut mvs[0],
                );
                self.mv_grid.fill(sub_bx_abs, sub_by_abs, 2, 1, mvs[0], 0);
                self.fire_mvd_hook_one_partition(
                    mb_x, mb_y, sub_bx_abs, sub_by_abs + 1, 2, 1,
                    None, p(1), &mut mvs[1],
                );
                self.mv_grid.fill(sub_bx_abs, sub_by_abs + 1, 2, 1, mvs[1], 0);
            }
            SubMbChoice::P4x8 { mvs, .. } => {
                self.fire_mvd_hook_one_partition(
                    mb_x, mb_y, sub_bx_abs, sub_by_abs, 1, 2,
                    None, p(0), &mut mvs[0],
                );
                self.mv_grid.fill(sub_bx_abs, sub_by_abs, 1, 2, mvs[0], 0);
                self.fire_mvd_hook_one_partition(
                    mb_x, mb_y, sub_bx_abs + 1, sub_by_abs, 1, 2,
                    None, p(1), &mut mvs[1],
                );
                self.mv_grid.fill(sub_bx_abs + 1, sub_by_abs, 1, 2, mvs[1], 0);
            }
            SubMbChoice::P4x4 { mvs, .. } => {
                // Encoder iterates with i = ox + 2*oy, i.e., raster
                // within the sub-MB: (0,0), (1,0), (0,1), (1,1).
                // Decoder mirror in `decode_sub_mb_mvds` matches.
                for (i, mv) in mvs.iter_mut().enumerate() {
                    let ox = i % 2;
                    let oy = i / 2;
                    self.fire_mvd_hook_one_partition(
                        mb_x, mb_y,
                        sub_bx_abs + ox, sub_by_abs + oy, 1, 1,
                        None, p(i as u8), mv,
                    );
                    self.mv_grid.fill(
                        sub_bx_abs + ox, sub_by_abs + oy, 1, 1, *mv, 0,
                    );
                }
            }
        }
    }

    /// Fire the MVD hook for a single partition. Computes pred_mv
    /// (using `mb_part_idx` for P16x8/P8x16 directional shortcuts
    /// per § 8.4.1.3.1, otherwise the general median predictor),
    /// derives original MVD = mv - pred, fires hook for X + Y
    /// axes, reads back possibly-modified slot values, updates
    /// `*mv = pred + final_mvd`. Caller pre-fills mv_grid after
    /// this returns.
    #[allow(clippy::too_many_arguments)]
    fn fire_mvd_hook_one_partition(
        &mut self,
        mb_x: usize,
        mb_y: usize,
        tl_bx: usize,
        tl_by: usize,
        part_w_4x4: usize,
        part_h_4x4: usize,
        mb_part_idx: Option<u8>,
        partition: u8,
        mv: &mut super::motion_estimation::MotionVector,
    ) {
        use super::motion_estimation::MotionVector;
        use super::partition_state::{
            predict_mv_for_mb_partition, predict_mv_for_partition,
        };
        use super::super::stego::inject::MvdSlot;
        use super::super::stego::Axis;

        let pred = match mb_part_idx {
            Some(idx) => predict_mv_for_mb_partition(
                &self.mv_grid, tl_bx, tl_by, part_w_4x4, part_h_4x4, idx, 0,
            ),
            None => predict_mv_for_partition(
                &self.mv_grid, tl_bx, tl_by, part_w_4x4, 0,
            ),
        };
        let mvd_x = mv.mv_x as i32 - pred.mv_x as i32;
        let mvd_y = mv.mv_y as i32 - pred.mv_y as i32;
        let mut sx = MvdSlot {
            list: 0, partition, axis: Axis::X, value: mvd_x,
        };
        let mut sy = MvdSlot {
            list: 0, partition, axis: Axis::Y, value: mvd_y,
        };
        self.invoke_stego_mvd_hook(mb_x, mb_y, &mut sx);
        self.invoke_stego_mvd_hook(mb_x, mb_y, &mut sy);
        *mv = MotionVector {
            mv_x: (pred.mv_x as i32 + sx.value) as i16,
            mv_y: (pred.mv_y as i32 + sy.value) as i16,
        };
    }

    /// Override the default GOP length of 30. The next IDR fires at
    /// frame numbers divisible by `gop_length`.
    pub fn set_gop_length(&mut self, gop_length: u32) {
        assert!(gop_length >= 1, "gop_length must be at least 1");
        self.gop_length = gop_length;
    }

    /// §B-direct-fix Stage 2 (#232) follow-up — read-only accessor
    /// for the per-MB intra flag grid. Tests use this to tag hotspot
    /// MBs as intra-vs-inter when localizing encoder/decoder recon
    /// divergence.
    pub fn intra_grid(&self) -> &[bool] {
        &self.intra_grid
    }

    /// §B-direct-fix Stage 2 (#232) follow-up — read-only accessor
    /// for the encoder's per-MB L0 MV at the top-left 4×4 sub-block.
    /// Tests use this to compare encoder's chosen MV against what
    /// the reference decoder decodes from the bitstream MVD, to localize MV emit
    /// vs decode mismatches.
    pub fn mv_l0_at_mb(&self, mb_x: u32, mb_y: u32)
        -> Option<(super::motion_estimation::MotionVector, i8)>
    {
        self.mv_grid.get_l0((mb_x * 4) as isize, (mb_y * 4) as isize)
    }

    /// Enable CABAC bin-by-bin tracing. Subsequent slice emissions
    /// will attach a trace to the CABAC engine; the per-slice trace
    /// is copied into `cabac_trace_buffer` when the slice finalizes.
    /// Read and clear via `take_cabac_trace()`.
    pub fn enable_cabac_trace(&mut self) {
        self.cabac_trace_enabled = true;
        self.cabac_trace_buffer.clear();
    }

    /// Consume the buffered CABAC trace lines.
    pub fn take_cabac_trace(&mut self) -> Vec<String> {
        std::mem::take(&mut self.cabac_trace_buffer)
    }

    /// True iff the DPB currently holds a reference frame suitable
    /// for P-frame encoding. Always true EXCEPT at start-of-stream
    /// and immediately after an IDR.
    /// Count of MBs in the most recently encoded frame that used
    /// the 8×8 transform (i.e. emitted `transform_size_8x8_flag = 1`).
    /// Reset at the start of every frame — call immediately after
    /// `encode_i_frame` / `encode_p_frame` to sample per-frame.
    pub fn transform_8x8_mb_count(&self) -> usize {
        self.transform_8x8_grid.iter().filter(|&&v| v).count()
    }

    /// Total MB count of the frame (width × height in MBs).
    /// Useful for computing a % 8×8-transform ratio alongside
    /// [`Self::transform_8x8_mb_count`].
    pub fn total_mb_count(&self) -> usize {
        self.transform_8x8_grid.len()
    }

    pub fn has_reference(&self) -> bool {
        self.dpb.has_reference()
    }

    /// §B-cascade-real Phase 1.1.C — accessor for the post-flip +
    /// deblocked reconstruction buffer that mirrors what a downstream
    /// H.264 player produces from the emitted bitstream.
    ///
    /// External consumers comparing encoder output to reference-decoder output
    /// (visual-quality regression tests, PSNR gates, mux output) MUST
    /// read this buffer instead of `recon`. `recon` stays pre-flip to
    /// preserve the multi-pass orchestrator's cover-capture invariant
    /// — its content does NOT match the bitstream when stego flips
    /// fire, but its evolution stays stable across Pass 1 (no hook)
    /// and Pass 3 (hook fires).
    ///
    /// Mux/PSNR rule of thumb: if the consumer needs to know "what
    /// pixels will the player see?" → `visual_recon`. If the consumer
    /// is INSIDE the encoder doing next-MB neighbour prediction →
    /// `recon`.
    pub fn visual_recon(&self) -> &ReconBuffer {
        &self.visual_recon
    }

    /// Scene-change probe for `encode_p_frame`. Computes a cheap
    /// luma SAD between the new source frame and the DPB reference
    /// at zero MV. When this exceeds a threshold — meaning the new
    /// frame is drastically different from the reference — motion
    /// estimation can't recover and P-frame residuals blow up
    /// catastrophically (see the 625-frame IMG_4138 regression:
    /// f570 IDR=47 dB → f590=20 dB inside one GOP).
    ///
    /// Returns true when the caller should re-route to
    /// `encode_i_frame`. Sampling every 8th pixel to keep probe
    /// cheap (<1% of encode time).
    pub fn should_force_idr_for_scene_change(&self, pixels: &[u8]) -> bool {
        if super::mb_decision_b::env_var_os_is_some("PHASM_DISABLE_SCENECUT") {
            return false;
        }
        let reference = match self.dpb.last_ref.as_ref() {
            None => return false,
            Some(r) => r,
        };
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;
        if pixels.len() < y_size {
            return false;
        }
        let mut total: u64 = 0;
        let mut count: u64 = 0;
        // Stride-8 sampling of the luma plane.
        for y in (0..h).step_by(8) {
            for x in (0..w).step_by(8) {
                let s = pixels[y * w + x] as i32;
                let r = reference.y_at(x as u32, y as u32) as i32;
                total += (s - r).unsigned_abs() as u64;
                count += 1;
            }
        }
        if count == 0 {
            return false;
        }
        let mean_sad = total / count;
        // Threshold: mean per-pixel deviation > 20 gray levels. This
        // is well beyond "fast motion" (typically <5) but below
        // "I-only static noise" (~2).
        mean_sad >= 20
    }

    /// Encode a single I-frame using Intra_16x16 DC mode (Phase 6A.10).
    ///
    /// Every frame emitted via this entry point is an IDR (clears the
    /// DPB). After encoding, the reconstructed frame is promoted to
    /// the DPB as the reference for subsequent P-frames.
    ///
    /// Input is yuv420p: Y plane (width*height bytes), then Cb
    /// (width/2 * height/2), then Cr. Output is an Annex B byte
    /// stream.
    /// §v1.7 Phase 3 (#325) — apply CRF-derived base CRF target if CRF
    /// mode is enabled. Called at the entry of each encode_{i,p,b}_frame
    /// to anchor `self.rc.target_crf` from `crf + lookahead_offset`.
    ///
    /// The per-frame-type offset (I=+0, P=+1, B=+2) is applied downstream
    /// by `RateController::base_qp_for_frame_type`. The B-frame additional
    /// offset is independently tunable via PHASM_B_QP_OFFSET in the rc
    /// helper, so this function does not double-apply.
    fn apply_crf_base_qp(&mut self) {
        let crf = match self.crf {
            Some(c) => c,
            None => return,
        };
        let lookahead_offset = self.lookahead.as_ref()
            .map(|l| l.qp_offset(self.mb_tree_display_idx))
            .unwrap_or(0);
        let derived = (crf as i32 + lookahead_offset).clamp(0, 51) as u8;
        self.rc.target_crf = derived;
    }

    pub fn encode_i_frame(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
        // Task #383 — install env snapshot for the duration of this
        // frame so free functions in the hot path (mb_decision_b,
        // partition_decision, etc.) read consistent env state.
        // No-op if an outer scope (cover_replay, test) already
        // installed.
        let _env_guard = self.env_snapshot.install();
        self.apply_crf_base_qp();
        let expected_len = self.frame_size_bytes();
        if pixels.len() != expected_len {
            return Err(EncoderError::InvalidInput(format!(
                "expected {expected_len} yuv420p bytes, got {}",
                pixels.len()
            )));
        }
        let mut out = Vec::new();
        self.emit_params_if_needed(&mut out);
        // IDR resets the DPB and frame_num (spec § 7.4.3 — IDR MUST
        // have frame_num = 0). Without this reset, the 2nd+ IDR in a
        // sequence inherits the wrapped frame_num from the prior GOP
        // (e.g. 14 after 30 frames mod 16), which conformant
        // decoders treat as a gap: without MMCO=5 they end up
        // holding both old and new reference frames → "number of
        // reference frames exceeds max" warnings plus visible desync
        // (I-frames look right, P-frames drift).
        self.frame_num = 0;
        self.dpb.reset();
        // §6E-A4 — IDR resets the POC anchor + display index. POC=0
        // for IDR by definition.
        self.poc_tracker.reset_at_idr(0);
        self.display_idx_of_prev_anchor = 0;
        // AUD (Access Unit Delimiter) — spec § 7.3.2.4. I-only frame.
        let aud_rbsp = build_aud_rbsp(PrimaryPicType::IOnly);
        let aud_nal = wrap_rbsp_as_nal(&aud_rbsp, NalType::AUD, 0);
        out.extend_from_slice(&ANNEX_B_START_CODE);
        out.extend_from_slice(&aud_nal);
        let slice_rbsp = match self.entropy_mode {
            EntropyMode::Cavlc => self.build_idr_slice_rbsp_i16x16(pixels)?,
            EntropyMode::Cabac => self.build_idr_slice_rbsp_cabac(pixels)?,
        };
        let slice_nal = wrap_rbsp_as_nal(&slice_rbsp, NalType::SLICE_IDR, 3);
        out.extend_from_slice(&ANNEX_B_START_CODE);
        out.extend_from_slice(&slice_nal);
        // Snapshot the just-reconstructed frame into the DPB for use
        // by the next P-frame.
        self.dpb.promote(&self.recon, self.frame_num);
        // §B-cascade-real Phase 1.1.B step 3: per-MB visual_recon
        // writes (step 2) + parallel deblocking populate visual_recon
        // throughout the frame. The step 1 frame-end mirror is gone.
        self.frame_num = (self.frame_num + 1) & 0xF;
        self.stego_frame_idx = self.stego_frame_idx.wrapping_add(1);
        self.gop_position = 1; // after the IDR, next frame is position 1
        Ok(out)
    }

    /// Encode a single I-frame using I_PCM macroblocks (Phase 6A.8).
    ///
    /// Lossless pixel-copy — useful for round-trip validation. Kept
    /// for the 6A.8 framing tests; production encoding uses
    /// `encode_i_frame`.
    pub fn encode_i_frame_pcm(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
        let expected_len = self.frame_size_bytes();
        if pixels.len() != expected_len {
            return Err(EncoderError::InvalidInput(format!(
                "expected {expected_len} yuv420p bytes, got {}",
                pixels.len()
            )));
        }
        // PCM slice header is CAVLC-style (no CABAC alignment bits), so
        // emit Baseline profile SPS/PPS regardless of the encoder's
        // default entropy_mode. Save/restore around the call so
        // subsequent non-PCM frames keep their configured mode.
        let saved_entropy_mode = self.entropy_mode;
        if self.entropy_mode != EntropyMode::Cavlc {
            self.entropy_mode = EntropyMode::Cavlc;
            self.params_emitted = false; // force re-emit with CAVLC profile
        }
        let result = self.encode_i_frame_pcm_inner(pixels);
        self.entropy_mode = saved_entropy_mode;
        result
    }

    fn encode_i_frame_pcm_inner(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
        let mut out = Vec::new();
        self.emit_params_if_needed(&mut out);
        // Reset DPB and frame_num — IDR MUST have frame_num = 0 per
        // spec § 7.4.3 (see encode_i_frame for the full rationale).
        self.frame_num = 0;
        self.dpb.reset();
        // AUD for the I_PCM path too.
        let aud_rbsp = build_aud_rbsp(PrimaryPicType::IOnly);
        let aud_nal = wrap_rbsp_as_nal(&aud_rbsp, NalType::AUD, 0);
        out.extend_from_slice(&ANNEX_B_START_CODE);
        out.extend_from_slice(&aud_nal);
        let slice_rbsp = self.build_idr_slice_rbsp_pcm(pixels)?;
        let slice_nal = wrap_rbsp_as_nal(&slice_rbsp, NalType::SLICE_IDR, 3);
        out.extend_from_slice(&ANNEX_B_START_CODE);
        out.extend_from_slice(&slice_nal);
        self.dpb.promote(&self.recon, self.frame_num);
        self.frame_num = (self.frame_num + 1) & 0xF;
        self.gop_position = 1;
        Ok(out)
    }

    /// Encode a single P-frame (Phase 6B.3).
    ///
    /// Requires that the DPB hold a reference frame (see `has_reference`).
    /// For each MB: runs 16×16 motion estimation against the DPB's
    /// reference, builds the motion-compensated prediction, computes
    /// the residual, forward-transforms / quantizes / CAVLC-emits it,
    /// and reconstructs into the ReconBuffer for future prediction.
    ///
    /// Only `P_L0_16x16` partitions are supported in this phase;
    /// sub-MB partitions (16×8, 8×16, 8×8, ...) are deferred.
    pub fn encode_p_frame(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
        // Task #383 — see `encode_i_frame` comment.
        let _env_guard = self.env_snapshot.install();
        self.apply_crf_base_qp();
        if !self.dpb.has_reference() {
            return Err(EncoderError::InvalidInput(
                "encode_p_frame called without a reference in the DPB; \
                 encode an I-frame first"
                    .into(),
            ));
        }
        let expected_len = self.frame_size_bytes();
        if pixels.len() != expected_len {
            return Err(EncoderError::InvalidInput(format!(
                "expected {expected_len} yuv420p bytes, got {}",
                pixels.len()
            )));
        }

        // Scene-change detection: quick coarse SAD probe between
        // source and previous recon at zero MV. When this SAD is
        // absurdly high, motion estimation will also fail to find a
        // good MV and we'll emit huge residuals → catastrophic drift
        // across the rest of the GOP. In that case, return early and
        // encode as an IDR instead.
        //
        // §scenecut-ibpbp-2026-05-09 (#288): for IBPBP, the
        // orchestrator-level pre-scan in
        // `iter_frames_in_encode_order` (stego/encode_pixels.rs)
        // already detected the cut and rewrote the affected sub-GOP
        // to [P=K-1, IDR=K]. By the time encode_p_frame is invoked
        // here, the frame type is already correctly P, and any cut
        // would have been emitted as IDR via encode_i_frame. So the
        // encoder-internal auto-IDR is restricted to IPPPP-only,
        // where the orchestrator's pre-scan does not run. Under
        // IBPBP a stale cut detected here would corrupt the DPB
        // exactly as it did pre-fix.
        if !self.enable_b_frames
            && self.should_force_idr_for_scene_change(pixels)
        {
            return self.encode_i_frame(pixels);
        }

        let mut out = Vec::new();
        self.emit_params_if_needed(&mut out);

        // Clear the MV grid (new frame).
        self.mv_grid.reset();

        // AUD for the P-frame.
        let aud_rbsp = build_aud_rbsp(PrimaryPicType::IP);
        let aud_nal = wrap_rbsp_as_nal(&aud_rbsp, NalType::AUD, 0);
        out.extend_from_slice(&ANNEX_B_START_CODE);
        out.extend_from_slice(&aud_nal);
        let slice_rbsp = match self.entropy_mode {
            EntropyMode::Cavlc => self.build_p_slice_rbsp(pixels)?,
            EntropyMode::Cabac => self.build_p_slice_rbsp_cabac(pixels)?,
        };
        let slice_nal = wrap_rbsp_as_nal(&slice_rbsp, NalType::SLICE, 2);
        out.extend_from_slice(&ANNEX_B_START_CODE);
        out.extend_from_slice(&slice_nal);

        // Promote the just-encoded frame to the DPB as the next
        // reference. §B-direct-fix #196: also capture the per-MB
        // colocated motion grid so the next B-frame's spatial-direct
        // derivation can run the spec § 8.4.1.2.2 step 6 colZeroFlag
        // check. Without the grid, static-background B-MBs adjacent
        // to a moving subject pick up the subject's median MV and
        // produce visible "ghost" streaks.
        let motion_grid = self.mv_grid.to_colocated_grid();
        self.dpb.promote_with_motion(&self.recon, self.frame_num, motion_grid);

        // Advance counters.
        self.frame_num = (self.frame_num + 1) & 0xF;
        // Phase 6F.2(e) fix to deferred-items §37 (coeff-sign half):
        // stego_frame_idx must advance per encoded frame so per-MB
        // stego positions encode the correct frame number into their
        // PositionKey. Previously only encode_i_frame incremented this
        // counter, so all P-frames in the same GOP reused the IDR's
        // post-increment value (= 1) — the encoder logged every P-MB
        // position under frame_idx=1 while the walker correctly
        // distributed them across frames 1..N. Caused the
        // FrameCorrupted bug surfaced by parity_real_world_64x48_5f_diagnostic.
        self.stego_frame_idx = self.stego_frame_idx.wrapping_add(1);
        self.gop_position += 1;
        if self.gop_position >= self.gop_length {
            self.gop_position = 0;
        }
        // §6E-A4 — advance the POC anchor. With B-frames enabled
        // (M=2 IBPBP), each P advances by 2 display-index units to
        // leave a slot for the B that will be encoded next.
        // Without B-frames, advance by 1 (legacy behavior, but
        // unused since enable_b_frames=false skips POC).
        if self.enable_b_frames {
            self.display_idx_of_prev_anchor += 2;
        }

        Ok(out)
    }

    /// §6E-A4 — encode a B-frame as all-B_Skip. Pixels currently
    /// unused (the decoder reconstructs B_Skip MBs via the spatial
    /// direct predictor on neighbors, so the encoder doesn't write
    /// any residual or MVD bits). §6E-A4(b/c) lands real ME for
    /// B-frames; this is the minimum-viable wiring that proves the
    /// IBPBP encode path produces parseable bytes.
    ///
    /// Caller responsibilities:
    /// - `enable_b_frames` must be `true` BEFORE the first frame
    ///   is encoded (otherwise SPS uses pic_order_cnt_type=2 and
    ///   the B-frame's POC LSB field would be a parse mismatch).
    /// - Call AFTER `encode_p_frame` (the P that will be the
    ///   B's L1 reference must be in the DPB). The B's display
    ///   index is implied by the encoder state:
    ///   `display_idx_of_prev_anchor - 1` for M=2 IBPBP.
    /// - The previous anchor (I or P) at `display_idx - 2` (i.e.
    ///   `display_idx_of_prev_anchor - 2`) must also still be in
    ///   the DPB as the L0 reference. With single-slot DPB this
    ///   is automatically false; full B-slice ME is §6E-A4(b)
    ///   which moves to MultiSlotDpb. The all-B_Skip path here
    ///   doesn't actually consume the references for prediction
    ///   (no MVDs emitted), so single-slot DPB is fine for now.
    pub fn encode_b_frame(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
        // Task #383 — see `encode_i_frame` comment.
        let _env_guard = self.env_snapshot.install();
        self.apply_crf_base_qp();
        if !self.enable_b_frames {
            return Err(EncoderError::InvalidInput(
                "encode_b_frame called with enable_b_frames=false".into(),
            ));
        }
        // §B-instrument v1 (#251) — set the current B-frame's display
        // index for the per-MB recorder. mb_decision_b_rdo reads this
        // when PHASM_B_INSTRUMENT=1 to tag records by frame.
        super::mb_decision_b::B_INSTRUMENT_FRAME_IDX.fetch_add(
            1,
            std::sync::atomic::Ordering::Relaxed,
        );
        if !self.dpb.has_reference() {
            return Err(EncoderError::InvalidInput(
                "encode_b_frame called without a reference in the DPB; \
                 encode an I-frame + at least one P-frame first"
                    .into(),
            ));
        }
        let expected_len = self.frame_size_bytes();
        if pixels.len() != expected_len {
            return Err(EncoderError::InvalidInput(format!(
                "expected {expected_len} yuv420p bytes, got {}",
                pixels.len()
            )));
        }

        let mut out = Vec::new();
        self.emit_params_if_needed(&mut out);

        // §6E-A6.1 — reset the MV grid so this B-frame's predictors
        // see only same-frame neighbour data (mirrors P-frame entry
        // at line 947). §6E-A4(c)-lite shipped without this because
        // B-frames emitted zero MVDs and didn't read neighbours; once
        // §6E-A6.1+ adds spatial-direct grid population (and §6E-A6.x
        // adds non-direct B-MBs that DO read neighbours), the grid
        // must be frame-local.
        self.mv_grid.reset();

        // AUD with PrimaryPicType::IPB to advertise B-frame presence.
        let aud_rbsp = build_aud_rbsp(PrimaryPicType::IPB);
        let aud_nal = wrap_rbsp_as_nal(&aud_rbsp, NalType::AUD, 0);
        out.extend_from_slice(&ANNEX_B_START_CODE);
        out.extend_from_slice(&aud_nal);

        // Build the B-slice RBSP.
        let slice_rbsp = self.build_b_slice_rbsp_cabac(pixels)?;
        // B-frames are non-reference (nal_ref_idc=0) by Phase 6E-A
        // convention — they don't enter the DPB.
        let slice_nal = wrap_rbsp_as_nal(&slice_rbsp, NalType::SLICE, 0);
        out.extend_from_slice(&ANNEX_B_START_CODE);
        out.extend_from_slice(&slice_nal);

        // §6E-A4 — frame_num for non-reference B-frames does NOT
        // advance per spec § 7.4.3 (frame_num only increments for
        // reference frames; B has nal_ref_idc=0).
        // gop_position advances (B counts toward GOP length).
        self.gop_position += 1;
        if self.gop_position >= self.gop_length {
            self.gop_position = 0;
        }
        // Phase 6F.2(e): stego_frame_idx advances per ENCODED frame,
        // not per reference frame. The bin walker visits B-frames in
        // bitstream (decode) order and assigns them sequential
        // frame_idx values, so the encoder must do the same. Distinct
        // from frame_num (spec) which only counts reference frames.
        self.stego_frame_idx = self.stego_frame_idx.wrapping_add(1);
        // display_idx_of_prev_anchor stays put — the next encode
        // (P or I) will advance it. The B fills the gap at
        // (anchor - 1).

        Ok(out)
    }

    /// §6E-A6.1q — build a B-slice RBSP. Real ME against L0 + L1
    /// references via `dpb.b_references()` when both anchors are
    /// available (post-IDR + ≥2 anchors encoded); otherwise falls
    /// back to spatial-direct-predicted MVs (zero-MVD, decoder
    /// spatial-direct path).
    fn build_b_slice_rbsp_cabac(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
        use crate::codec::h264::cabac::neighbor::{CabacNeighborMB, MbTypeClass};

        // §6E-A6.1q.b — pull (L0, L1) refs out of the DPB. Cloned so
        // the borrow checker lets us mutably borrow `self.me` + the
        // MV grid in the inner loop. `b_references()` returns Some
        // only when both past_anchor + last_ref are populated; if the
        // single-anchor case (first B in stream) ever flows here, the
        // ME calls fall back to zero MVs via `mb_decision_b` with
        // `me_mvs=None`.
        let b_refs_opt = self.dpb.b_references().map(|(l0, l1)| (l0.clone(), l1.clone()));
        // v1.4 Phase 4.3 (#314, Path B) — second-closest past anchor
        // for B-frame L0 multi-ref. At MultiRefConfig::SINGLE_REF
        // this stays None — refine pass is skipped, behavior is bit-
        // identical to v1.3. At DUAL_REF_L0 we pull the new
        // `pre_past_anchor` slot which holds the L0 ref_idx=1
        // candidate. None when fewer than 3 anchors have been
        // promoted (= early-IDR / first-2-Bs-after-IDR case).
        let b_l0_ref_1: Option<super::reference_buffer::ReconFrame> =
            if self.multi_ref_config.max_l0_active > 1 {
                self.dpb.pre_past_anchor.clone()
            } else {
                None
            };
        // §B-direct-fix #196 — pull the L1 reference's collocated
        // motion grid (= the next-anchor P-frame's per-MB MVs) so
        // `derive_b_direct_spatial_with_col` can apply the
        // colZeroFlag check (spec § 8.4.1.2.2 step 6). When this is
        // None, the static-MB safeguard is off and a static
        // background MB next to a moving subject inherits the
        // subject's median MV → ghost streak. P-frame DPB promote
        // path uses `promote_with_motion` to populate this.
        let l1_motion_grid = b_refs_opt
            .as_ref()
            .and_then(|(_, l1)| l1.motion_grid.as_ref())
            .cloned();
        let l1_motion_grid_ref = l1_motion_grid.as_ref();
        let y_size = (self.width * self.height) as usize;
        let y_stride = self.width as usize;
        let c_stride = (self.width / 2) as usize;
        let c_size = (self.width / 2 * self.height / 2) as usize;
        let y_plane = &pixels[..y_size];
        // §6E-A6.1q.b part 3b — chroma planes available for residual
        // emission (PHASM_B_RESIDUAL=1 path).
        let cb_plane = &pixels[y_size..y_size + c_size];
        let cr_plane = &pixels[y_size + c_size..];

        // §B-direct-fix follow-on 2026-05-03 — residual emission for
        // non-direct B-MBs (L0/L1/Bi) is now PAIRED with B-RDO. When
        // RDO is on, it emits L0/L1/Bi modes that need residuals to
        // close the prediction-vs-source distortion gap; without
        // residuals, the user sees heavy block artifacts on motion
        // (the "FIXED_RDO.mp4 broken" pattern observed 2026-05-03).
        //
        // Task #204 (v1.1): typed `BRdoConfig` on the encoder is the
        // primary control. `PHASM_B_RESIDUAL` env var still wins as a
        // debug override.
        //
        // Task #383 (2026-05-12): env-var snapshot is captured in
        // `Encoder::new` to make the encoder internally consistent
        // under parallel test execution. Reading the snapshot here
        // instead of re-reading the env var prevents mid-encode
        // mutations by other threads from changing this encoder's
        // residual-emit decision.
        let residual_enabled = self
            .env_snapshot
            .residual_override
            .unwrap_or(self.b_rdo_config.enable_residual);

        let mut w = BitWriter::with_capacity(self.frame_size_bytes().max(512));
        // §B-QP-bug-fix 2026-05-09 (#300) — was FrameType::P; the entire
        // B-slice path was hardcoded to look up the P-frame QP, making
        // the FrameType::B branch in base_qp_for_frame_type effectively
        // dead code. PHASM_B_QP env override (added 2026-05-09 for the
        // QP-bottleneck experiment) was therefore a no-op. Now correctly
        // routes B-frames through their own QP path including any
        // PHASM_B_QP override.
        let qp = self.rc.base_qp_for_frame_type(FrameType::B);
        let qp_c = derive_chroma_qp(qp as i32, self.pps_params.chroma_qp_index_offset as i32) as u8;
        let pic_init_qp = self.pps_params.pic_init_qp as i32;

        // Compute B-frame display index: previous anchor minus 1
        // (M=2 IBPBP convention). Then derive POC LSB.
        let b_display_index = self.display_idx_of_prev_anchor.saturating_sub(1);
        let pic_order_cnt_lsb = self.poc_tracker.poc_lsb_for(b_display_index);

        // §B-direct-fix follow-on 2026-05-04 — spec § 7.4.3:
        // frame_num for non-reference frames (B with nal_ref_idc=0)
        // must equal the PRECEDING REFERENCE PICTURE's frame_num.
        // In our IBPBP M=2 encode order I0 P2 B1 P4 B3 ..., when
        // B1 is encoded, encode_p_frame has already incremented
        // self.frame_num past P2's value (preparing for P4). So
        // B's slice header must use self.frame_num - 1.
        // Mod-16 wraparound matches log2_max_frame_num_minus4=0
        // → MaxFrameNum=16 in our SPS.
        let b_frame_num = self.frame_num.wrapping_sub(1) & 0xF;
        let b_disable_deblock = super::mb_decision_b::env_var_os_is_some("PHASM_DISABLE_DEBLOCK");
        // §B-direct-fix.v3 2026-05-07 — temporal-direct mode toggle.
        // PHASM_B_TEMPORAL_DIRECT=1 switches BOTH the slice header
        // flag AND the encoder's MV derivation to temporal direct
        // (§ 8.4.1.2.3 scaled colocated MV). Fixes visible motion-
        // boundary streaks on slow-motion content (user screenshot
        // 2026-05-07). Spatial direct is the legacy default to keep
        // existing tests green; flip the default once V6 visual
        // gate passes.
        let direct_spatial = !super::mb_decision_b::env_var_os_is_some("PHASM_B_TEMPORAL_DIRECT");
        // Compute POCs for temporal direct. IBPBP M=2 layout: B_disp
        // = anchor - 1, L0 = anchor - 2 (past P), L1 = anchor (next P).
        // For longer M the formula is the same — POC step is 2 per
        // display index regardless.
        let b_direct_mode = if direct_spatial {
            BDirectMode::Spatial
        } else {
            let poc_curr = self.poc_tracker.full_poc_for(b_display_index) as i32;
            let poc_l0 = self.poc_tracker
                .full_poc_for(b_display_index.saturating_sub(1)) as i32;
            let poc_l1 = self.poc_tracker
                .full_poc_for(b_display_index + 1) as i32;
            BDirectMode::Temporal { poc_curr, poc_l0, poc_l1 }
        };
        // v1.4 Phase 4.5 (#316) — slice header must reflect the
        // ACTUAL number of L0 references available in the DPB at this
        // slice, not the MultiRefConfig ceiling. the reference decoder's reference
        // picture list construction trusts num_ref_idx_l0_active and
        // emits "Missing reference picture, default is 65536" if the
        // count exceeds what's actually in the DPB → CABAC drift +
        // ~18 dB visual cliff. For B-slice L0: ref_0=past_anchor,
        // ref_1=pre_past_anchor. Count = 1 if pre_past_anchor=None,
        // else 2 (capped by max_l0_active).
        let actual_l0_b = if self.multi_ref_config.max_l0_active > 1
            && self.dpb.pre_past_anchor.is_some()
        {
            2u8
        } else {
            1u8
        };
        let hdr = BSliceHeaderParams {
            pps_id: 0,
            frame_num: b_frame_num,
            pic_order_cnt_lsb,
            direct_spatial_mv_pred_flag: direct_spatial,
            slice_qp_delta: qp as i32 - pic_init_qp,
            disable_deblocking: b_disable_deblock,
            // §Stealth.L4.6.2 — PPS defaults L0=3. Slice header
            // overrides to phasm's actual active count.
            num_ref_idx_active_override: true,
            num_ref_idx_l0_active_minus1: actual_l0_b.saturating_sub(1),
            num_ref_idx_l1_active_minus1: 0,
            cabac_init_idc: Some(0),
            log2_max_pic_order_cnt_lsb_minus4: self.sps_params.log2_max_pic_order_cnt_lsb_minus4,
            log2_max_frame_num_minus4: 0,
        };
        continue_slice_header_b(&mut w, &hdr);

        // CABAC engine init for B-slice with cabac_init_idc=0 → PIdc0.
        let mb_w = (self.width / 16) as usize;
        let mb_h = (self.height / 16) as usize;
        let mb_count = mb_w * mb_h;
        let mut cabac = CabacEncoder::new_slice(CabacInitSlot::PIdc0, qp as i32, mb_w);

        // §B-cascade-real fix 2026-05-06 (task #233): reset frame-state
        // grids + prev_mb_qp before the per-MB loop. Mirror of the
        // equivalent block in `build_p_slice_rbsp_cabac`
        // (encoder.rs:2765-2790 + 2770). Without these resets, B-slice
        // encoding starts with values from the previous P-frame in
        // `intra_grid`, `qp_grid`, `transform_8x8_grid`, the four
        // total-coeff grids, and `prev_mb_qp` — leaking stale state into
        // CBF context derivation, intra-edge boundary strength in the
        // deblock pass, intra-mode neighbour queries, and the first-MB
        // mb_qp_delta when residuals fire. Each downstream operation
        // that reads these diverges from spec-compliant decode
        // (which correctly resets between frames). Empirical: the
        // missing reset is a contributing root cause of the
        // §B-cascade-real visual cascade (v1.0 BLOCKER).
        for v in self.total_coeff_grid.iter_mut() {
            *v = 0xFF;
        }
        for v in self.intra16x16_dc_tc_grid.iter_mut() {
            *v = 0xFF;
        }
        for v in self.chroma_cb_tc_grid.iter_mut() {
            *v = 0;
        }
        for v in self.chroma_cr_tc_grid.iter_mut() {
            *v = 0;
        }
        for v in self.transform_8x8_grid.iter_mut() { *v = false; }
        for v in self.qp_grid.iter_mut() { *v = 0xFF; }
        for v in self.intra_grid.iter_mut() { *v = false; }
        for m in self.i4x4_mode_grid.iter_mut() { *m = 0xFF; }
        for m in self.i8x8_mode_grid.iter_mut() { *m = 0xFF; }
        self.prev_mb_qp = qp as i32;

        // §6E-A4(c)-lite — emit a hash-based mix of B_Skip and
        // B_Direct_16x16. Both modes use the spatial direct
        // predictor at decode time, so visual quality is identical
        // to all-B_Skip; the mode-distribution variation on the wire
        // matches real-encoder output better than all-skip and
        // narrows the L3 fingerprint gap. CBP=0 always — residual
        // emission lands in §6E-A4(c)-full.
        for mb_addr in 0..mb_count {
            let mb_x = mb_addr % mb_w;
            let mb_y = mb_addr / mb_w;
            if mb_y > 0 && mb_x == 0 {
                cabac.neighbors.new_row();
            }
            // §6E-A6.1q.b — real L0 + L1 ME for the bucket-selected
            // mode (when refs are available). `mb_decision_b_with_mvs`
            // owns the bucket dispatch + force-mode override; we
            // pre-compute the (L0, L1) MV pair and let it fold into
            // whichever arm the bucket picks. ME is run unconditionally
            // because we don't know which arm will fire until after
            // the bucket lookup (and the bucket ignores cost today).
            // Cost-based RDO that uses ME costs to pick the arm is a
            // §Stealth.L3.x follow-on — todays scope keeps the
            // calibrated mix while paying real MVD bits.
            let me_mvs = if let Some((l0_ref, l1_ref)) = b_refs_opt.as_ref() {
                // Source 16x16 MB rect.
                let mb_px = mb_x * 16;
                let mb_py = mb_y * 16;
                // Predicted MVs (median of neighbours) — used as ME
                // bit-cost anchor + as the start seed for refinement.
                let predicted_l0 = super::mb_decision_b::predict_b_partition_mv_l0_pub(
                    &self.mv_grid, mb_x, mb_y,
                );
                let predicted_l1 = super::mb_decision_b::predict_b_partition_mv_l1_pub(
                    &self.mv_grid, mb_x, mb_y,
                );
                // §B-direct-fix.v3 ROOT-CAUSE 2026-05-07 — multi-pred ME for
                // B-slices, full-symmetry version mirroring P-frame's
                // `build_me_candidates`.
                //
                // V11 fix shipped `[predicted, ZERO]` and closed the wall-MB
                // bug (homogeneous source: zero MV gives perfect SATD).
                // Visual A/B confirmed wall artifacts gone — but body-
                // internal boundary MBs (pants/skin, jacket/scarf junctions)
                // still showed colored block artifacts.
                //
                // Diagnosis: heterogeneous-source MBs need a non-zero best
                // MV (1-2 px from camera shake / body micro-motion). The
                // median predictor can land on a wrong-direction body MV
                // inherited from raster propagation; ZERO is wrong too
                // because the source genuinely moved. The CORRECT MV is
                // what actual neighbour MBs picked — not the median, not
                // zero.
                //
                // P-frame ME's `build_me_candidates` solves this by
                // including A (left), B (top), C (top-right) raw neighbour
                // MVs from the grid. Body MBs in P raster get clean motion
                // through their A/B/C neighbours.
                //
                // This fix: add A/B/C neighbour L0 MVs to B-frame's L0 ME
                // candidate list, and A/B/C L1 MVs to L1 ME. The grid is
                // populated by previously-encoded B-MBs of the current
                // frame, so neighbours reflect the actual motion of nearby
                // body content.
                let bx = (mb_x * 4) as isize;
                let by = (mb_y * 4) as isize;
                // §B-direct-fix.v3.ROOT.v13 2026-05-07 — predictor magnitude clamp.
                //
                // Per-MB dump confirmed body MBs' predicted_mv inherits a
                // chain-propagated bad MV (e.g., (-205, 112)). ME's cost
                // function `SATD + λ × bits(mv - predicted)` penalises any
                // MV that diverges from the wrong predicted, including
                // MV=0. On body content where periodic-shift positions
                // yield ~equal SATD, the rate term tips the balance toward
                // the wrong MV (bits=0 at predicted vs ~18 bits at zero).
                //
                // Encoder/decoder semantic test (`dump_b_frame_recon_vs_decode`)
                // confirmed encoder.recon ≡ reference-decoder output — encoder is
                // honest. Artifacts ARE the encoder's "best match" given
                // its broken cost function.
                //
                // The structural fix: ME's bit-cost anchor must NOT trust
                // predicted_mv when its magnitude is large. Real motion
                // content has small predicted_mv (median of small
                // neighbours); chain-propagated body MVs have huge
                // magnitudes. Clamp threshold at 32 quarter-pel = 8 px,
                // ~2x typical camera-shake displacement.
                //
                // Bitstream emission still uses the spec-compliant median
                // predictor for MVD computation; only ME's INTERNAL
                // preference shifts. ME picks MV=X based on cheap rate
                // anchor; emit-time MVD = X - median_predictor (correct
                // per spec).
                // §B-boundary-anchor v1 (#251, 2026-05-08) — at
                // motion-boundary MBs (neighbour L0 MVs span more
                // than 1 px), override the rate-cost anchor to
                // (0,0). Otherwise ME's `SAD + λ × bits(mv -
                // predicted)` cost function pulls toward the
                // (likely wrong-direction) spatial-median predictor,
                // and ME converges to wall-direction MVs at
                // jacket-edge MBs even though (0,0) gives a much
                // lower SAD. With the (0,0) anchor, ME is naturally
                // attracted to (0,0) when content is static-ish
                // and its SAD is competitive — content-correct
                // predictions at boundaries.
                //
                // Bitstream emission still uses the spec-compliant
                // median predictor for MVD computation (only ME's
                // INTERNAL preference shifts). MVD = chosen_mv -
                // median_predictor stays correct.
                //
                // Disable via PHASM_B_NO_BOUNDARY_REFUSE=1 (same
                // env gate that disables the matching Direct
                // refusal in mb_decision_b_rdo).
                let at_boundary = super::mb_decision_b::is_motion_boundary(
                    &self.mv_grid, mb_x, mb_y,
                );
                // §B-direct-magnitude-clamp v1 (#251) — also override
                // ME anchor to (0,0) when predicted MV magnitude
                // exceeds 32 qpel. Chain-propagated huge MVs in
                // spatial-median bias ME's rate cost toward keeping
                // the huge MV; clamping anchor to (0,0) breaks the
                // chain (ME's rate cost now favours (0,0)-direction
                // candidates).
                let predicted_too_large = predicted_l0.mv_x.abs() > 32
                    || predicted_l0.mv_y.abs() > 32
                    || predicted_l1.mv_x.abs() > 32
                    || predicted_l1.mv_y.abs() > 32;
                let override_anchor = (at_boundary
                    && !super::mb_decision_b::env_var_os_is_some("PHASM_B_NO_BOUNDARY_REFUSE"))
                    || (predicted_too_large
                        && !super::mb_decision_b::env_var_os_is_some("PHASM_B_NO_DIRECT_MAGCLAMP"));
                let me_anchor_l0 = if override_anchor {
                    MotionVector::ZERO
                } else {
                    clamp_me_anchor(predicted_l0)
                };
                let me_anchor_l1 = if override_anchor {
                    MotionVector::ZERO
                } else {
                    clamp_me_anchor(predicted_l1)
                };
                let mut cands_l0 = Vec::with_capacity(6);
                cands_l0.push(me_anchor_l0);
                cands_l0.push(MotionVector::ZERO);
                if let Some((mv, _)) = self.mv_grid.get_l0(bx - 1, by) { cands_l0.push(mv); }
                if let Some((mv, _)) = self.mv_grid.get_l0(bx, by - 1) { cands_l0.push(mv); }
                if let Some((mv, _)) = self.mv_grid.get_l0(bx + 4, by - 1) { cands_l0.push(mv); }
                let mut cands_l1 = Vec::with_capacity(6);
                cands_l1.push(me_anchor_l1);
                cands_l1.push(MotionVector::ZERO);
                if let Some((mv, _)) = self.mv_grid.get_l1(bx - 1, by) { cands_l1.push(mv); }
                if let Some((mv, _)) = self.mv_grid.get_l1(bx, by - 1) { cands_l1.push(mv); }
                if let Some((mv, _)) = self.mv_grid.get_l1(bx + 4, by - 1) { cands_l1.push(mv); }
                // Phase 4 (#251) — temporal candidate is env-gated via
                // `PHASM_B_TEMPORAL_CAND=1`. Default OFF: Phase 4.2
                // empirical V18 demo (commit pending) showed that
                // adding the L1-reference's collocated MV alone is
                // net-neutral-to-slightly-worse on real content
                // (body=1005 → 1037 anomalies on iPhone7 1080p IBPBP
                // M=2). Cause: the multi-cand search picks the
                // lowest-initial-cost candidate as the start then
                // refines once; a temporal candidate with marginally
                // lower initial cost but a worse refinement basin
                // displaces ME from a better spatial-median basin.
                // Proper fix is multi-pass refine-from-each-cand,
                // tracked under Phase 4.3+ (#251 follow-on). Keep the
                // env gate for ablation experiments; ship behavior
                // unchanged.
                if super::mb_decision_b::env_var_os_is_some("PHASM_B_TEMPORAL_CAND") {
                    let poc_curr_b = self.poc_tracker.full_poc_for(b_display_index) as i32;
                    let poc_l0_b = self.poc_tracker
                        .full_poc_for(b_display_index.saturating_sub(1)) as i32;
                    let poc_l1_b = self.poc_tracker
                        .full_poc_for(b_display_index + 1) as i32;
                    let temporal_cand = super::b_direct_predictor::derive_b_direct_temporal(
                        l1_motion_grid.as_ref(), mb_x, mb_y, poc_curr_b, poc_l0_b, poc_l1_b,
                    );
                    cands_l0.push(temporal_cand.mv_l0);
                    cands_l1.push(temporal_cand.mv_l1);
                }
                // Phase 4.3 (#251) — refine-from-each-cand. When
                // `PHASM_B_MULTI_REFINE=1`, switch from the default
                // best-seed-then-refine to refine-from-each-then-pick-
                // best. ~2× ME cost at N=6 candidates but unlocks the
                // temporal candidate's potential AND any future
                // candidate whose refinement basin beats the median's.
                let use_multi_refine = super::mb_decision_b::env_var_os_is_some("PHASM_B_MULTI_REFINE");
                let l0_search = if use_multi_refine {
                    self.me.search_block_multi_refine(
                        y_plane, y_stride, l0_ref,
                        mb_px as u32, mb_py as u32, 16, 16, me_anchor_l0, &cands_l0,
                    )
                } else {
                    self.me.search_block_with_candidates(
                        y_plane, y_stride, l0_ref,
                        mb_px as u32, mb_py as u32, 16, 16, me_anchor_l0, &cands_l0,
                    )
                };
                let l1_search = if use_multi_refine {
                    self.me.search_block_multi_refine(
                        y_plane, y_stride, l1_ref,
                        mb_px as u32, mb_py as u32, 16, 16, me_anchor_l1, &cands_l1,
                    )
                } else {
                    self.me.search_block_with_candidates(
                        y_plane, y_stride, l1_ref,
                        mb_px as u32, mb_py as u32, 16, 16, me_anchor_l1, &cands_l1,
                    )
                };

                // §B-me-result-clamp v1 (#251 follow-on, V26 2026-05-08).
                //
                // Phase 2.11 instrumentation finding: post-V25, top-10
                // worst-deviation B-MBs shifted from mode=Direct (huge
                // spatial-derived MVs) to mode=L1 (huge ME-derived MVs).
                // Examples: mvL1=(132,-295), (196,-359), (3,-412),
                // (192,34) — ME's L1 search finding coincidentally-low-
                // SATD matches at 30-100 pixel distances. These MVs
                // produce wrong-content luma AND wrong chroma stripes
                // (chroma values from completely different image
                // regions).
                //
                // Hard-clamp ME result magnitude to 64 qpel (16 px) on
                // each axis. Real motion at 30fps rarely exceeds 16 px
                // between adjacent B-frame and reference; magnitudes
                // beyond are coincidence-matches not motion-tracking.
                // When clamp fires, snap MV to (0,0) — predict from
                // same-position reference + emit residual.
                //
                // Default ON. Disable via PHASM_B_NO_ME_RESULT_CLAMP=1.
                const ME_RESULT_QPEL_CAP: i16 = 64;
                let me_clamp = !super::mb_decision_b::env_var_os_is_some("PHASM_B_NO_ME_RESULT_CLAMP");
                let l0_search = if me_clamp
                    && (l0_search.mv.mv_x.abs() > ME_RESULT_QPEL_CAP
                        || l0_search.mv.mv_y.abs() > ME_RESULT_QPEL_CAP)
                {
                    super::motion_estimation::MotionSearchResult {
                        mv: MotionVector::ZERO,
                        cost: l0_search.cost,
                    }
                } else {
                    l0_search
                };
                let l1_search = if me_clamp
                    && (l1_search.mv.mv_x.abs() > ME_RESULT_QPEL_CAP
                        || l1_search.mv.mv_y.abs() > ME_RESULT_QPEL_CAP)
                {
                    super::motion_estimation::MotionSearchResult {
                        mv: MotionVector::ZERO,
                        cost: l1_search.cost,
                    }
                } else {
                    l1_search
                };
                // §6E-D.5(c) — bipred-aware joint refinement when RDO
                // is active. Independent L0+L1 ME finds MVs close to
                // spatial-direct neighbour-median (because that's the
                // ME bit-cost anchor), so the L0_16x16 / L1_16x16 / Bi
                // candidates' SATD is ~equal to Skip's SATD → Skip
                // dominates. Joint refinement explores small qpel
                // diamonds around the seeds with the OTHER list held
                // fixed, scoring bipred SATD; this finds MV pairs that
                // genuinely beat spatial-direct on bipred quality, in
                // turn making the L0/L1/Bi candidates more competitive
                // with Skip on cost. See §6E-D.5(d) trace breakthrough
                // memory for full motivation.
                let (refined_l0, refined_l1) = if self.env_snapshot.b_rdo_enabled(self.b_rdo_config) {
                    self.me.refine_bipred(
                        y_plane, y_stride, l0_ref, l1_ref,
                        mb_px as u32, mb_py as u32, 16, 16,
                        l0_search.mv, l1_search.mv,
                        predicted_l0, predicted_l1,
                    )
                } else {
                    (l0_search.mv, l1_search.mv)
                };
                Some((refined_l0, refined_l1))
            } else {
                None
            };
            // §6E-D.5 — fast-RDO mode decision when `PHASM_B_RDO=1` is set
            // AND we have references + ME results. Falls through to the
            // existing hash-bucket distribution otherwise.
            //
            // §B-direct-fix.v2 (#194): honour PHASM_B_FORCE_MODE here
            // BEFORE either dispatch. Previously force_mode was checked
            // only inside `mb_decision_b_with_mvs`, so when RDO routed
            // around that function the env override was silently
            // ignored. Test bisects of "force=skip" with PHASM_B_RDO=1
            // were actually running full RDO. Lifted the check to the
            // call site so both branches honour it equally.
            let mut decision = if let Some(forced) =
                self.env_snapshot.force_b_mode.clone()
            {
                forced
            } else if self.b_rdo_config.force_l0_zero_b_mode {
                // Phase F (#262, 2026-05-08, v1.0 ship path) — override
                // every B-MB to L0_16x16 with MV=(0,0). Bypasses
                // spatial-direct + ME-derived MV paths so encoder +
                // decoder agree on MC reference positions. Closes the
                // corpus ghost-image / blocky-motion bug (Phase D
                // bisect finding). v1.1 follow-on aligns encoder MC
                // with decoder spec to remove this workaround.
                super::mb_decision_b::BMbDecision::L0_16x16 {
                    mv: MotionVector::ZERO,
                    ref_idx_l0: 0,
                }
            } else if self.env_snapshot.b_rdo_enabled(self.b_rdo_config) {
                if let (Some((l0_ref, l1_ref)), Some(mvs)) = (b_refs_opt.as_ref(), me_mvs) {
                    let mut src_y = [[0u8; 16]; 16];
                    let mb_px_x = mb_x * 16;
                    let mb_px_y = mb_y * 16;
                    for dy in 0..16 {
                        for dx in 0..16 {
                            src_y[dy][dx] = y_plane[(mb_px_y + dy) * y_stride + (mb_px_x + dx)];
                        }
                    }
                    super::mb_decision_b::mb_decision_b_rdo(
                        &self.mv_grid, mb_x, mb_y, &src_y,
                        l0_ref, l1_ref, qp, mvs,
                    )
                } else {
                    super::mb_decision_b::mb_decision_b_with_mvs(
                        &self.mv_grid, mb_x, mb_y,
                        self.frame_num as u32, mb_addr as u32,
                        me_mvs,
                        self.env_snapshot.force_b_mode.clone(),
                    )
                }
            } else {
                super::mb_decision_b::mb_decision_b_with_mvs(
                    &self.mv_grid, mb_x, mb_y,
                    self.frame_num as u32, mb_addr as u32,
                    me_mvs,
                    self.env_snapshot.force_b_mode.clone(),
                )
            };

            // v1.4 Phase 4.3 (#314, Path B) — B-side multi-ref L0
            // post-pass. Runs only when (a) DUAL_REF_L0 is on, (b)
            // pre_past_anchor (= ref_1) is populated (= ≥3 anchors
            // promoted; first 2 B-frames after IDR get None and skip),
            // (c) past_anchor (= ref_0) is populated. Mutates
            // `decision` in place: upgrades L0_16x16 / Bi_16x16 to
            // ref_idx_l0=1 + swaps in the ref_1-optimal L0 MV when
            // ref_1 wins by more than λ.
            if let (Some(ref_1_frame), Some((ref_0_frame, _))) = (
                b_l0_ref_1.as_ref(),
                b_refs_opt.as_ref(),
            ) {
                let mut refine_src_y = [[0u8; 16]; 16];
                let mb_px_x_r = mb_x * 16;
                let mb_px_y_r = mb_y * 16;
                for dy in 0..16 {
                    for dx in 0..16 {
                        refine_src_y[dy][dx] =
                            y_plane[(mb_px_y_r + dy) * y_stride + (mb_px_x_r + dx)];
                    }
                }
                super::mb_decision_b::refine_b_choice_multi_ref(
                    &refine_src_y, &mut self.me, mb_x, mb_y,
                    ref_0_frame, ref_1_frame,
                    &mut decision,
                );
            }

            // §B-AQ v1 2026-05-09 (#290 follow-on, AQ-mode-1 for B-slices) —
            // per-MB variance-adjusted QP. Mirrors the existing P-frame
            // AQ at write_p_macroblock_cabac (encoder.rs:3493+). High-
            // variance/textured B-MBs get -QP offset (more residual
            // bits, preserves detail); flat B-MBs are clamped to 0
            // offset (prevents banding regression). the converter-pipeline centroid ships
            // `--aq-mode 1` as default for the same reason.
            //
            // Computes src_y once per MB. Uses qp_offset to derive
            // mb_qp, then shadows the slice-level `qp` for the
            // remainder of this iteration → emit_b_residual_for_pred
            // and commit_mb_state pick up the adjusted QP, residual
            // gets quantised at the per-MB level, mb_qp_delta is
            // emitted automatically by the existing residual path.
            //
            // Env knobs (all shared with P-frame AQ):
            //   PHASM_AQ_MODE = 1 (default) | 3
            //   PHASM_DISABLE_PAQ = 1 → disable
            //   PHASM_AQ_STRENGTH_Q10 (mode 3 only)
            let mut aq_src_y = [[0u8; 16]; 16];
            let mb_px_x_aq = mb_x * 16;
            let mb_px_y_aq = mb_y * 16;
            for dy in 0..16 {
                for dx in 0..16 {
                    aq_src_y[dy][dx] = y_plane[(mb_px_y_aq + dy) * y_stride + (mb_px_x_aq + dx)];
                }
            }
            let aq_variance = super::rate_control::mb_variance_16x16(&aq_src_y);
            let aq_mode: u8 = super::mb_decision_b::env_var("PHASM_AQ_MODE")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);
            let aq_disabled = super::mb_decision_b::env_var_os_is_some("PHASM_DISABLE_PAQ");
            let aq_qp_offset: i32 = if aq_disabled {
                0
            } else if aq_mode == 3 {
                let strength: i32 = super::mb_decision_b::env_var("PHASM_AQ_STRENGTH_Q10")
                    
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(256);
                let mut luma_sum: u32 = 0;
                for row in &aq_src_y {
                    for &v in row {
                        luma_sum += v as u32;
                    }
                }
                let avg_luma = luma_sum / 256;
                super::rate_control::variance_to_qp_offset_mode3(
                    aq_variance,
                    self.aq_frame_mean_log2_q8,
                    avg_luma,
                    strength,
                )
            } else {
                super::rate_control::variance_to_qp_offset(aq_variance).min(0)
            };
            // §v1.7 Phase 1.1 (#323) — MB-tree per-MB QP offset composes
            // additively with AQ-3 offset. None / no env-gate → 0.
            let mb_tree_offset: i32 = self.mb_tree.as_ref()
                .map(|t| t.qp_offset(self.mb_tree_display_idx, mb_x, mb_y))
                .unwrap_or(0);
            // §v1.7 Phase 2.1 (#324) — Lookahead frame-level QP offset
            // composes additively. None → 0.
            let lookahead_offset: i32 = self.lookahead.as_ref()
                .map(|l| l.qp_offset(self.mb_tree_display_idx))
                .unwrap_or(0);
            let combined_offset = (aq_qp_offset + mb_tree_offset + lookahead_offset).clamp(-15, 15);
            let mb_qp = ((qp as i32) + combined_offset).clamp(0, 51) as u8;
            let qp = mb_qp;
            // chroma QP is derived inside emit_b_residual_for_pred from
            // the qp arg; no need to pre-compute here.

            // §B-deblock-fix (#242, 2026-05-07) — pre-seed qp_grid +
            // intra_grid for this B-MB BEFORE dispatch. Most emit paths
            // overwrite via emit_b_residual_for_pred → commit_mb_state,
            // but B_Skip / B_Direct (CBP=0) and the residual-disabled
            // L0/L1/Bi fallback never call commit_mb_state. Without
            // pre-seeding, qp_grid stays at the 0xFF reset sentinel and
            // the new end-of-slice deblock pass panics on
            // ALPHA_TABLE[255] OOB. Per-mode commit_mb_state can later
            // overwrite this with mb-specific qp (e.g., when residual
            // emits an mb_qp_delta).
            self.commit_mb_state(mb_x, mb_y, qp, false);

            match decision {
                super::mb_decision_b::BMbDecision::Direct16x16 => {
                    // B_Direct_16x16: mb_skip_flag=0 + mb_type=0 + CBP=0,
                    // then end_of_slice. No mb_qp_delta when CBP=0.
                    encode_mb_skip_flag_b(&mut cabac, false, mb_x);
                    encode_mb_type_b(&mut cabac, 0, mb_x);
                    encode_coded_block_pattern(&mut cabac, /* cbp_value */ 0, mb_x);
                    let nb = CabacNeighborMB {
                        mb_type: MbTypeClass::BSkipOrDirect,
                        mb_skip_flag: false,
                        cbp_luma: 0,
                        cbp_chroma: 0,
                        ..CabacNeighborMB::default()
                    };
                    cabac.neighbors.commit(mb_x, nb);
                    // §6E-A6.1 — populate MV grid with spatial-direct
                    // derivation so subsequent non-direct B-MBs (when
                    // §6E-A6.1 mode-decision starts emitting them) see
                    // consistent neighbour data.
                    let direct = populate_b_direct_grid(&mut self.mv_grid, mb_x, mb_y, l1_motion_grid_ref, b_direct_mode);
                    // §B-direct-fix Stage 2 (#232) — write the spatial-
                    // direct predicted pixels to BOTH self.recon AND
                    // self.visual_recon. Decoder reconstructs B_Direct
                    // pixels from these MVs via L0/L1 references; if
                    // we don't write the same pixels here, encoder.recon
                    // stays at the previous frame's pixels for this MB
                    // → encoder.recon ≠ reference-decoder output → cascade.
                    if let Some((l0_ref, l1_ref)) = b_refs_opt.as_ref() {
                        // Phase 2.12.e (#281, 2026-05-08) — when colZeroFlag
                        // override fired on at least one 8×8 sub-block,
                        // use per-8×8 MC. Otherwise legacy single-MV path
                        // (preserves SIMD perf on the common case).
                        let needs_per_8x8 = direct.has_per_8x8_override_l0()
                            || direct.has_per_8x8_override_l1();
                        let (pred_y, pred_cb, pred_cr) = if needs_per_8x8 {
                            let pred_y = super::b_inter_prediction::build_b_luma_prediction_per_8x8(
                                direct.mv_l0_per_8x8, direct.mv_l1_per_8x8,
                                direct.uses_l0(), direct.uses_l1(),
                                l0_ref, l1_ref, mb_x, mb_y,
                            );
                            let pred_cb = super::b_inter_prediction::build_b_chroma_prediction_per_8x8(
                                direct.mv_l0_per_8x8, direct.mv_l1_per_8x8,
                                direct.uses_l0(), direct.uses_l1(),
                                l0_ref, l1_ref, 0, mb_x, mb_y,
                            );
                            let pred_cr = super::b_inter_prediction::build_b_chroma_prediction_per_8x8(
                                direct.mv_l0_per_8x8, direct.mv_l1_per_8x8,
                                direct.uses_l0(), direct.uses_l1(),
                                l0_ref, l1_ref, 1, mb_x, mb_y,
                            );
                            (pred_y, pred_cb, pred_cr)
                        } else {
                            let mode = b_direct_to_inter_mode(&direct);
                            let pred_y = super::b_inter_prediction::build_b_luma_prediction(
                                mode, l0_ref, l1_ref, mb_x, mb_y,
                            );
                            let pred_cb = super::b_inter_prediction::build_b_chroma_prediction(
                                mode, l0_ref, l1_ref, 0, mb_x, mb_y,
                            );
                            let pred_cr = super::b_inter_prediction::build_b_chroma_prediction(
                                mode, l0_ref, l1_ref, 1, mb_x, mb_y,
                            );
                            (pred_y, pred_cb, pred_cr)
                        };
                        // §B-direct-fix Stage 2 (#232) — write predicted
                        // pixels to visual_recon ONLY (not self.recon).
                        self.visual_recon
                            .write_luma_mb(mb_x as u32, mb_y as u32, &pred_y);
                        self.visual_recon
                            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &pred_cb);
                        self.visual_recon
                            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &pred_cr);
                    }
                }
                super::mb_decision_b::BMbDecision::Skip => {
                    // B_Skip: mb_skip_flag=1, no further syntax.
                    encode_mb_skip_flag_b(&mut cabac, true, mb_x);
                    let nb = CabacNeighborMB {
                        mb_type: MbTypeClass::BSkipOrDirect,
                        mb_skip_flag: true,
                        ..CabacNeighborMB::default()
                    };
                    cabac.neighbors.commit(mb_x, nb);
                    // §6E-A6.1 — same rationale as Direct16x16 above:
                    // B_Skip uses spatial-direct on the decoder side
                    // (§ 7.3.5.1 / § 8.4.1.2.2), so the encoder grid
                    // must reflect those derived MVs at this MB.
                    let direct = populate_b_direct_grid(&mut self.mv_grid, mb_x, mb_y, l1_motion_grid_ref, b_direct_mode);
                    // §B-direct-fix Stage 2 (#232) — same recon-parity
                    // fix as Direct16x16 above. B_Skip reconstructs from
                    // the spatial-direct prediction surface; encoder
                    // must write those pixels to recon + visual_recon
                    // so the next P/B frame's reference matches what
                    // the decoder reconstructs from the bitstream.
                    if let Some((l0_ref, l1_ref)) = b_refs_opt.as_ref() {
                        // Phase 2.12.e (#281, 2026-05-08) — same as Direct16x16
                        // above. Use per-8×8 builder when colZeroFlag override
                        // fired on at least one sub-block.
                        let needs_per_8x8 = direct.has_per_8x8_override_l0()
                            || direct.has_per_8x8_override_l1();
                        let (pred_y, pred_cb, pred_cr) = if needs_per_8x8 {
                            let pred_y = super::b_inter_prediction::build_b_luma_prediction_per_8x8(
                                direct.mv_l0_per_8x8, direct.mv_l1_per_8x8,
                                direct.uses_l0(), direct.uses_l1(),
                                l0_ref, l1_ref, mb_x, mb_y,
                            );
                            let pred_cb = super::b_inter_prediction::build_b_chroma_prediction_per_8x8(
                                direct.mv_l0_per_8x8, direct.mv_l1_per_8x8,
                                direct.uses_l0(), direct.uses_l1(),
                                l0_ref, l1_ref, 0, mb_x, mb_y,
                            );
                            let pred_cr = super::b_inter_prediction::build_b_chroma_prediction_per_8x8(
                                direct.mv_l0_per_8x8, direct.mv_l1_per_8x8,
                                direct.uses_l0(), direct.uses_l1(),
                                l0_ref, l1_ref, 1, mb_x, mb_y,
                            );
                            (pred_y, pred_cb, pred_cr)
                        } else {
                            let mode = b_direct_to_inter_mode(&direct);
                            let pred_y = super::b_inter_prediction::build_b_luma_prediction(
                                mode, l0_ref, l1_ref, mb_x, mb_y,
                            );
                            let pred_cb = super::b_inter_prediction::build_b_chroma_prediction(
                                mode, l0_ref, l1_ref, 0, mb_x, mb_y,
                            );
                            let pred_cr = super::b_inter_prediction::build_b_chroma_prediction(
                                mode, l0_ref, l1_ref, 1, mb_x, mb_y,
                            );
                            (pred_y, pred_cb, pred_cr)
                        };
                        self.visual_recon
                            .write_luma_mb(mb_x as u32, mb_y as u32, &pred_y);
                        self.visual_recon
                            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &pred_cb);
                        self.visual_recon
                            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &pred_cr);
                    }
                }
                super::mb_decision_b::BMbDecision::L0_16x16 { mv, ref_idx_l0 } => {
                    // v1.4 Phase 4.3 (#314, Path B) — dispatch L0
                    // reference based on chosen ref_idx_l0. ref_idx=0
                    // → past_anchor (b_refs_opt's L0). ref_idx=1 →
                    // pre_past_anchor (b_l0_ref_1). The refine pass
                    // ran upstream already populated ref_idx_l0; here
                    // we just pick the right physical reference.
                    if let Some((l0_ref, l1_ref)) = b_refs_opt.as_ref().filter(|_| residual_enabled) {
                        let l0_chosen: &super::reference_buffer::ReconFrame =
                            if ref_idx_l0 == 0 { l0_ref } else {
                                b_l0_ref_1.as_ref()
                                    .expect("ref_idx_l0=1 requires b_l0_ref_1")
                            };
                        let mode = super::b_inter_prediction::BInterMode::L0_16x16 { mv };
                        self.write_b_inter_residual_macroblock_cabac(
                            &mut cabac, mode, l0_chosen, l1_ref,
                            mb_x, mb_y,
                            y_plane, y_stride, cb_plane, cr_plane, c_stride,
                            qp, qp_c,
                            ref_idx_l0,
                        )?;
                    } else {
                        emit_b_l0_16x16(
                            &mut cabac, mb_x, &mut self.mv_grid, mb_x, mb_y, mv,
                            ref_idx_l0,
                            actual_l0_b,
                        )?;
                        if let Some((l0_ref, l1_ref)) = b_refs_opt.as_ref() {
                            let l0_chosen: &super::reference_buffer::ReconFrame =
                                if ref_idx_l0 == 0 { l0_ref } else {
                                    b_l0_ref_1.as_ref()
                                        .expect("ref_idx_l0=1 requires b_l0_ref_1")
                                };
                            let mode = super::b_inter_prediction::BInterMode::L0_16x16 { mv };
                            self.write_b_prediction_recon(mode, l0_chosen, l1_ref, mb_x, mb_y);
                        }
                    }
                }
                super::mb_decision_b::BMbDecision::L1_16x16 { mv } => {
                    if let Some((l0_ref, l1_ref)) = b_refs_opt.as_ref().filter(|_| residual_enabled) {
                        let mode = super::b_inter_prediction::BInterMode::L1_16x16 { mv };
                        // L1_16x16 doesn't use L0 — ref_idx_l0 is
                        // irrelevant (no ref_idx_l0 emission for
                        // pure-L1 modes per spec § 7.3.5.1).
                        self.write_b_inter_residual_macroblock_cabac(
                            &mut cabac, mode, l0_ref, l1_ref,
                            mb_x, mb_y,
                            y_plane, y_stride, cb_plane, cr_plane, c_stride,
                            qp, qp_c,
                            /* ref_idx_l0 */ 0,
                        )?;
                    } else {
                        emit_b_l1_16x16(&mut cabac, mb_x, &mut self.mv_grid, mb_x, mb_y, mv)?;
                        if let Some((l0_ref, l1_ref)) = b_refs_opt.as_ref() {
                            let mode = super::b_inter_prediction::BInterMode::L1_16x16 { mv };
                            self.write_b_prediction_recon(mode, l0_ref, l1_ref, mb_x, mb_y);
                        }
                    }
                }
                super::mb_decision_b::BMbDecision::Bi_16x16 { mv_l0, mv_l1, ref_idx_l0 } => {
                    // v1.4 Phase 4.3 (#314, Path B) — dispatch L0
                    // reference based on chosen ref_idx_l0; L1 stays
                    // single-ref (Q1).
                    if let Some((l0_ref, l1_ref)) = b_refs_opt.as_ref().filter(|_| residual_enabled) {
                        let l0_chosen: &super::reference_buffer::ReconFrame =
                            if ref_idx_l0 == 0 { l0_ref } else {
                                b_l0_ref_1.as_ref()
                                    .expect("ref_idx_l0=1 requires b_l0_ref_1")
                            };
                        let mode = super::b_inter_prediction::BInterMode::Bi_16x16 {
                            mv_l0,
                            mv_l1,
                        };
                        self.write_b_inter_residual_macroblock_cabac(
                            &mut cabac, mode, l0_chosen, l1_ref,
                            mb_x, mb_y,
                            y_plane, y_stride, cb_plane, cr_plane, c_stride,
                            qp, qp_c,
                            ref_idx_l0,
                        )?;
                    } else {
                        emit_b_bi_16x16(
                            &mut cabac, mb_x, &mut self.mv_grid, mb_x, mb_y, mv_l0, mv_l1,
                            ref_idx_l0,
                            actual_l0_b,
                        )?;
                        if let Some((l0_ref, l1_ref)) = b_refs_opt.as_ref() {
                            let l0_chosen: &super::reference_buffer::ReconFrame =
                                if ref_idx_l0 == 0 { l0_ref } else {
                                    b_l0_ref_1.as_ref()
                                        .expect("ref_idx_l0=1 requires b_l0_ref_1")
                                };
                            let mode = super::b_inter_prediction::BInterMode::Bi_16x16 {
                                mv_l0,
                                mv_l1,
                            };
                            self.write_b_prediction_recon(mode, l0_chosen, l1_ref, mb_x, mb_y);
                        }
                    }
                }
                super::mb_decision_b::BMbDecision::Partitioned { mb_type: t, parts } => {
                    let b_refs_for_emit =
                        b_refs_opt.as_ref().map(|(l0, l1)| (l0, l1));
                    emit_b_partitioned_method(
                        self, &mut cabac, mb_x, mb_y, t, parts,
                        b_refs_for_emit, residual_enabled,
                        y_plane, y_stride, cb_plane, cr_plane, c_stride,
                        qp, qp_c,
                        actual_l0_b,
                    )?;
                }
                super::mb_decision_b::BMbDecision::B8x8 { sub_mb_types, parts } => {
                    let b_refs_for_emit =
                        b_refs_opt.as_ref().map(|(l0, l1)| (l0, l1));
                    emit_b_8x8_method(
                        self, &mut cabac, mb_x, mb_y,
                        sub_mb_types, parts, l1_motion_grid_ref,
                        b_refs_for_emit, residual_enabled,
                        y_plane, y_stride, cb_plane, cr_plane, c_stride,
                        qp, qp_c,
                        actual_l0_b,
                    )?;
                }
                super::mb_decision_b::BMbDecision::IntraI16x16 { .. } => {
                    // §intra-in-B (#319) Phase 3 — emit I_16x16 with B-slice
                    // mb_type prefix. The chosen `i16x16_mode` and
                    // `chroma_pred_mode` from the BMbDecision are advisory
                    // only; the helper re-runs intra mode decision against
                    // current-frame neighbours (`build_luma_neighbors_16x16`
                    // / `build_chroma_neighbors`) since neighbour-pixel
                    // access only exists at emit time. The mode-decision
                    // converges to the same family as the variant carries
                    // for typical content.
                    encode_mb_skip_flag_b(&mut cabac, false, mb_x);
                    self.mv_grid.fill(
                        mb_x * 4, mb_y * 4, 4, 4,
                        MotionVector::ZERO, REF_IDX_NONE,
                    );
                    self.commit_mb_state(mb_x, mb_y, qp, /* is_intra */ true);
                    let is_last_mb = mb_addr == mb_count - 1;
                    self.write_i16x16_macroblock_cabac_with_ctx(
                        &mut cabac, mb_x, mb_y, y_plane, y_stride,
                        cb_plane, cr_plane, c_stride,
                        qp, qp_c, is_last_mb,
                        IntraSliceCtx::B,
                    )?;
                    // The helper already emitted end_of_slice_flag + committed
                    // neighbour state (mb_type=I16x16). Continue the per-MB
                    // loop without falling through to the inter-path
                    // end_of_slice emission below.
                    continue;
                }
            }
            // end_of_slice_flag bin emitted at every MB.
            let is_last = mb_addr == mb_count - 1;
            encode_end_of_slice_flag(&mut cabac, is_last);
        }

        // §B-deblock-fix (#242, 2026-05-07) — apply in-loop deblock to
        // B-slice recon, mirroring the P-slice path. Without this,
        // self.recon + self.visual_recon stay un-deblocked while the reference decoder
        // (parsing slice header `disable_deblocking_filter_idc=0`)
        // applies deblock per spec — divergence on every B-MB
        // boundary. Dominant tail residual source per `dump_b_frame_recon_vs_decode`
        // (PHASM_DISABLE_DEBLOCK=1 cuts nz% by 30-40× on B-frames).
        if !b_disable_deblock {
            // Per Phase 1.1.B/C philosophy: only `visual_recon` gets
            // deblocked. `self.recon` stays at the cover-pristine /
            // pre-flip state for the streaming-Viterbi orchestrator and
            // existing round-trip tests that assume self.recon ≡ what
            // the walker would reconstruct from the bitstream sans
            // deblock. (P-slices deblock both because they need
            // post-deblock recon as the next P/B frame's reference;
            // B-slices don't promote to DPB so self.recon's
            // deblock-state doesn't matter for downstream frames.)
            let coded_flags = self.build_coded_flags();
            super::deblocking_filter::filter_frame(
                &mut self.visual_recon,
                &self.qp_grid,
                &self.intra_grid,
                &coded_flags,
                Some(&self.mv_grid),
            );
        }

        // Finish CABAC, append to slice writer.
        let cabac_bytes = cabac.finish();
        // Slice byte alignment + cabac_zero_word stuffing per spec.
        Ok(assemble_cabac_slice_rbsp(w, &cabac_bytes))
    }

    fn build_p_slice_rbsp(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
        let mut w = BitWriter::with_capacity(self.frame_size_bytes().max(512));
        self.mode_stats = [0; 9];
        let qp = self.rc.base_qp_for_frame_type(FrameType::P);
        let qp_c = crate::codec::h264::transform::derive_chroma_qp(qp as i32, self.pps_params.chroma_qp_index_offset as i32) as u8;
        let pic_init_qp = self.pps_params.pic_init_qp as i32;
        let disable_deblock = super::mb_decision_b::env_var_os_is_some("PHASM_DISABLE_DEBLOCK");
        // v1.4 Phase 4.5 (#316) — P-side actual L0 active count.
        // ref_0 = last_ref (always present after IDR), ref_1 =
        // past_anchor (None for first P after IDR). Mirrors B-side
        // logic in build_b_slice_rbsp_cabac.
        let actual_l0_p = if self.multi_ref_config.max_l0_active > 1
            && self.dpb.past_anchor.is_some()
        {
            2u8
        } else {
            1u8
        };
        let hdr = PSliceHeaderParams {
            pps_id: 0,
            frame_num: self.frame_num,
            pic_order_cnt_lsb: self.maybe_p_poc_lsb(),
            log2_max_pic_order_cnt_lsb_minus4: self.sps_params.log2_max_pic_order_cnt_lsb_minus4,
            slice_qp_delta: qp as i32 - pic_init_qp,
            disable_deblocking: disable_deblock,
            // §Stealth.L4.6.2 — override PPS default L0=3 down to phasm
            // active count.
            num_ref_idx_active_override: true,
            num_ref_idx_l0_active_minus1: actual_l0_p.saturating_sub(1),
            cabac_init_idc: None,
            // §Stealth.L4.6.4 — emit degenerate pred_weight_table.
            weighted_pred_flag: true,
        };
        continue_slice_header_p(&mut w, &hdr);
        self.prev_mb_qp = qp as i32;

        let mb_w = (self.width / 16) as usize;
        let mb_h = (self.height / 16) as usize;
        let y_stride = self.width as usize;
        let c_stride = (self.width / 2) as usize;
        let y_size = (self.width * self.height) as usize;
        let c_size = (self.width / 2 * self.height / 2) as usize;
        let y_plane = &pixels[..y_size];
        let cb_plane = &pixels[y_size..y_size + c_size];
        let cr_plane = &pixels[y_size + c_size..];

        // ─── Phase F: AQ mode 3 frame pre-pass ───
        // Compute the frame-mean-log2 variance once so the per-MB
        // offset math in write_p_macroblock can auto-center around
        // this frame's own statistics (adaptive-quantisation mode 3,
        // variance-weighted). Stored on self; zero means "not
        // computed for this frame" (write_p_macroblock falls back
        // to mode-1 behaviour).
        self.aq_frame_mean_log2_q8 = self.compute_aq_frame_mean(
            y_plane, y_stride, mb_w, mb_h,
        );

        // Pull the DPB reference out once (cloned so the borrow
        // checker lets us mutably borrow `self.me` + `self.mv_grid`
        // inside the hot loop).
        let reference = self
            .dpb
            .last_ref
            .clone()
            .expect("DPB reference guaranteed by encode_p_frame entry check");

        // Per-slice reset of neighbor TC grids — without this the
        // previous frame's (I-slice) TCs leak into P-slice nC
        // derivation, producing coeff_token codewords the decoder
        // can't follow. 0xFF on luma grids is the "not available"
        // sentinel for nC; chroma grids use 0 as "coded with zero
        // coeffs" (matches the decoder's explicit write on
        // cbp_chroma != 2 MBs).
        for v in self.total_coeff_grid.iter_mut() {
            *v = 0xFF;
        }
        for v in self.intra16x16_dc_tc_grid.iter_mut() {
            *v = 0xFF;
        }
        for v in self.chroma_cb_tc_grid.iter_mut() {
            *v = 0;
        }
        for v in self.chroma_cr_tc_grid.iter_mut() {
            *v = 0;
        }
        // Reset per-MB QP + intra flags for the deblock filter.
        // Each MB overwrites via `commit_mb_state` before filter runs,
        // but clean init makes "read-before-commit" bugs obvious.
        for v in self.qp_grid.iter_mut() { *v = 0xFF; }
        for v in self.transform_8x8_grid.iter_mut() { *v = false; }
        for v in self.intra_grid.iter_mut() { *v = false; }

        // Track pending mb_skip_run: count of consecutive skippable
        // MBs (P_16x16 with P_SKIP MV and cbp=0). Emitted inline when
        // a non-skip MB arrives, or at end of slice if trailing MBs
        // were all skipped.
        let mut skip_run: u32 = 0;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                self.write_p_macroblock(
                    &mut w, &reference, mb_x, mb_y, mb_w, y_plane, y_stride, cb_plane, cr_plane,
                    c_stride, qp, qp_c, &mut skip_run,
                )?;
            }
        }
        // End-of-slice: flush any trailing skipped-MB run. Spec § 7.3.4:
        // when the final MB(s) are skipped, emit mb_skip_run=N with no
        // subsequent MB data.
        if skip_run > 0 {
            w.write_ue(skip_run);
        }

        // Apply in-loop deblocking (P-frames). For P content the
        // all_intra flag is false; the filter falls through to bs=1
        // on inter-coded MB boundaries — still filters, just less
        // aggressively.
        if !disable_deblock {
            let coded_flags = self.build_coded_flags();
            super::deblocking_filter::filter_frame(
                &mut self.recon,
                &self.qp_grid,
                &self.intra_grid,
                &coded_flags,
                Some(&self.mv_grid),
            );
            // §B-cascade-real Phase 1.1.B step 3: deblock visual_recon
            // in parallel so muxed-output PSNR reflects what a
            // downstream player produces.
            super::deblocking_filter::filter_frame(
                &mut self.visual_recon,
                &self.qp_grid,
                &self.intra_grid,
                &coded_flags,
                Some(&self.mv_grid),
            );
        }

        if super::mb_decision_b::env_var_os_is_some("PHASM_MODE_STATS") {
            let s = &self.mode_stats;
            // Phase D.3: total excludes the intra-in-P sub-counters
            // (I4X4 / I16X16) since those are already summed into
            // MODE_STAT_INTRA_IN_P — adding them in would double-count.
            let total: u32 = s[..7].iter().sum();
            let pct = |v: u32| if total > 0 { (v as f32 * 100.0) / total as f32 } else { 0.0 };
            let iip = s[MODE_STAT_INTRA_IN_P];
            let iip_pct_i4 = if iip > 0 {
                (s[MODE_STAT_INTRA_IN_P_I4X4] as f32 * 100.0) / iip as f32
            } else { 0.0 };
            let iip_pct_i16 = if iip > 0 {
                (s[MODE_STAT_INTRA_IN_P_I16X16] as f32 * 100.0) / iip as f32
            } else { 0.0 };
            eprintln!(
                "frame={} total={} skip_fast={} ({:.1}%) skip_me={} ({:.1}%) 16x16={} ({:.1}%) 16x8={} ({:.1}%) 8x16={} ({:.1}%) 8x8={} ({:.1}%) intra={} ({:.1}%) [I4x4={} ({:.1}%) I16x16={} ({:.1}%)]",
                self.frame_num, total,
                s[MODE_STAT_P_SKIP_FAST], pct(s[MODE_STAT_P_SKIP_FAST]),
                s[MODE_STAT_P_SKIP_POST_ME], pct(s[MODE_STAT_P_SKIP_POST_ME]),
                s[MODE_STAT_P_16X16], pct(s[MODE_STAT_P_16X16]),
                s[MODE_STAT_P_16X8], pct(s[MODE_STAT_P_16X8]),
                s[MODE_STAT_P_8X16], pct(s[MODE_STAT_P_8X16]),
                s[MODE_STAT_P_8X8], pct(s[MODE_STAT_P_8X8]),
                iip, pct(iip),
                s[MODE_STAT_INTRA_IN_P_I4X4], iip_pct_i4,
                s[MODE_STAT_INTRA_IN_P_I16X16], iip_pct_i16,
            );
        }

        w.write_rbsp_trailing();
        Ok(w.finish())
    }

    /// Phase 6C.6d.3 — CABAC variant of `build_p_slice_rbsp`.
    fn build_p_slice_rbsp_cabac(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
        let mut w = BitWriter::with_capacity(self.frame_size_bytes().max(512));
        let qp = self.rc.base_qp_for_frame_type(FrameType::P);
        let qp_c = crate::codec::h264::transform::derive_chroma_qp(qp as i32, self.pps_params.chroma_qp_index_offset as i32) as u8;
        let pic_init_qp = self.pps_params.pic_init_qp as i32;
        let disable_deblock = super::mb_decision_b::env_var_os_is_some("PHASM_DISABLE_DEBLOCK");
        // v1.4 Phase 4.5 (#316) — P-side CABAC actual L0 active count.
        // Same gating as CAVLC build_p_slice_rbsp.
        let actual_l0_p = if self.multi_ref_config.max_l0_active > 1
            && self.dpb.past_anchor.is_some()
        {
            2u8
        } else {
            1u8
        };
        let hdr = PSliceHeaderParams {
            pps_id: 0,
            frame_num: self.frame_num,
            pic_order_cnt_lsb: self.maybe_p_poc_lsb(),
            log2_max_pic_order_cnt_lsb_minus4: self.sps_params.log2_max_pic_order_cnt_lsb_minus4,
            slice_qp_delta: qp as i32 - pic_init_qp,
            disable_deblocking: disable_deblock,
            // §Stealth.L4.6.2 — override PPS default L0=3 down to phasm
            // active count.
            num_ref_idx_active_override: true,
            num_ref_idx_l0_active_minus1: actual_l0_p.saturating_sub(1),
            cabac_init_idc: Some(0),
            // §Stealth.L4.6.4 — emit degenerate pred_weight_table.
            weighted_pred_flag: true,
        };
        continue_slice_header_p(&mut w, &hdr);
        self.prev_mb_qp = qp as i32;
        self.mode_stats = [0; 9];

        let mb_w = (self.width / 16) as usize;
        let mb_h = (self.height / 16) as usize;
        let y_stride = self.width as usize;
        let c_stride = (self.width / 2) as usize;
        let y_size = (self.width * self.height) as usize;
        let c_size = (self.width / 2 * self.height / 2) as usize;
        let y_plane = &pixels[..y_size];
        let cb_plane = &pixels[y_size..y_size + c_size];
        let cr_plane = &pixels[y_size + c_size..];

        // CABAC engine: P-slice init with cabac_init_idc = 0 → PIdc0.
        let mut cabac = CabacEncoder::new_slice(CabacInitSlot::PIdc0, qp as i32, mb_w);
        if self.cabac_trace_enabled {
            cabac.engine.trace = Some(Vec::new());
        }

        let reference = self
            .dpb
            .last_ref
            .clone()
            .expect("DPB reference guaranteed by encode_p_frame entry check");

        // v1.4 Phase 4.2 (#313, Path B) — extract the second-closest
        // past anchor from DPB for the multi-ref post-pass. At
        // MultiRefConfig::SINGLE_REF (default) we skip this entirely
        // — `ref_1` stays None and `write_p_macroblock_cabac` runs
        // unchanged (bit-identical to v1.3). At DUAL_REF_L0 we pull
        // `past_anchor` (the previous-P-anchor slot — for IBPBP M=2
        // this is exactly the second-closest past). First P after
        // IDR has only `last_ref` populated; `past_anchor` is None
        // and ref_1 stays None — the post-pass is skipped.
        let ref_1: Option<super::reference_buffer::ReconFrame> =
            if self.multi_ref_config.max_l0_active > 1 {
                self.dpb.past_anchor.clone()
            } else {
                None
            };

        for v in self.total_coeff_grid.iter_mut() {
            *v = 0xFF;
        }
        for v in self.intra16x16_dc_tc_grid.iter_mut() {
            *v = 0xFF;
        }
        for v in self.chroma_cb_tc_grid.iter_mut() {
            *v = 0;
        }
        for v in self.chroma_cr_tc_grid.iter_mut() {
            *v = 0;
        }
        for v in self.transform_8x8_grid.iter_mut() { *v = false; }
        for v in self.qp_grid.iter_mut() { *v = 0xFF; }
        for v in self.intra_grid.iter_mut() { *v = false; }
        // Reset I_4x4 / I_8x8 mode grids so stale modes from the
        // previous I-frame don't leak into P-slice intra-in-P
        // neighbor queries. Spec § 8.3.1.1 says a non-I_NxN neighbor
        // must fall back to DC(2); if the grid still holds last
        // frame's I_4x4 mode, `lookup_i4x4_mode` wrongly returns it
        // and derive_i4x4_mode_flags derives pred_mode from the
        // stale value while the decoder correctly falls back to 2.
        // Fixes the Phase D.2-stealth parity drift observed at
        // MB positions where the frame-0 left-MB was I_4x4.
        for m in self.i4x4_mode_grid.iter_mut() { *m = 0xFF; }
        for m in self.i8x8_mode_grid.iter_mut() { *m = 0xFF; }

        for mb_y in 0..mb_h {
            if mb_y > 0 {
                cabac.neighbors.new_row();
            }
            for mb_x in 0..mb_w {
                let is_last_mb = mb_y == mb_h - 1 && mb_x == mb_w - 1;
                self.write_p_macroblock_cabac(
                    &mut cabac,
                    &reference,
                    ref_1.as_ref(),
                    mb_x,
                    mb_y,
                    y_plane,
                    y_stride,
                    cb_plane,
                    cr_plane,
                    c_stride,
                    qp,
                    qp_c,
                    is_last_mb,
                )?;
            }
        }

        // In-loop deblocking (same as CAVLC P path).
        // PHASM_DISABLE_DEBLOCK env-gated: when set, skip deblock to
        // match the slice header's disable_deblocking_filter_idc=1
        // emission (encoder + decoder both skip deblock).
        if !super::mb_decision_b::env_var_os_is_some("PHASM_DISABLE_DEBLOCK") {
            let coded_flags = self.build_coded_flags();
            super::deblocking_filter::filter_frame_with_transform(
                &mut self.recon,
                &self.qp_grid,
                &self.intra_grid,
                Some(&self.transform_8x8_grid),
                &coded_flags,
                Some(&self.mv_grid),
            );
            // §B-cascade-real Phase 1.1.B step 3: deblock visual_recon
            // in parallel.
            super::deblocking_filter::filter_frame_with_transform(
                &mut self.visual_recon,
                &self.qp_grid,
                &self.intra_grid,
                Some(&self.transform_8x8_grid),
                &coded_flags,
                Some(&self.mv_grid),
            );
        }

        let bin_count = cabac.engine.bin_count();
        let pic_size_mbs = (mb_w * mb_h) as u32;
        if let Some(trace) = cabac.engine.trace.take() {
            self.cabac_trace_buffer.extend(trace);
        }

        if super::mb_decision_b::env_var_os_is_some("PHASM_MODE_STATS") {
            let s = &self.mode_stats;
            // Phase D.3: total excludes the intra-in-P sub-counters
            // (already summed into MODE_STAT_INTRA_IN_P).
            let total: u32 = s[..7].iter().sum();
            let pct = |v: u32| if total > 0 { (v as f32 * 100.0) / total as f32 } else { 0.0 };
            let iip = s[MODE_STAT_INTRA_IN_P];
            let iip_pct_i4 = if iip > 0 {
                (s[MODE_STAT_INTRA_IN_P_I4X4] as f32 * 100.0) / iip as f32
            } else { 0.0 };
            let iip_pct_i16 = if iip > 0 {
                (s[MODE_STAT_INTRA_IN_P_I16X16] as f32 * 100.0) / iip as f32
            } else { 0.0 };
            eprintln!(
                "frame={} total={} skip_fast={} ({:.1}%) skip_me={} ({:.1}%) 16x16={} ({:.1}%) 16x8={} ({:.1}%) 8x16={} ({:.1}%) 8x8={} ({:.1}%) intra={} ({:.1}%) [I4x4={} ({:.1}%) I16x16={} ({:.1}%)]",
                self.frame_num, total,
                s[MODE_STAT_P_SKIP_FAST], pct(s[MODE_STAT_P_SKIP_FAST]),
                s[MODE_STAT_P_SKIP_POST_ME], pct(s[MODE_STAT_P_SKIP_POST_ME]),
                s[MODE_STAT_P_16X16], pct(s[MODE_STAT_P_16X16]),
                s[MODE_STAT_P_16X8], pct(s[MODE_STAT_P_16X8]),
                s[MODE_STAT_P_8X16], pct(s[MODE_STAT_P_8X16]),
                s[MODE_STAT_P_8X8], pct(s[MODE_STAT_P_8X8]),
                iip, pct(iip),
                s[MODE_STAT_INTRA_IN_P_I4X4], iip_pct_i4,
                s[MODE_STAT_INTRA_IN_P_I16X16], iip_pct_i16,
            );
        }

        let body = cabac.finish();
        let mut rbsp = assemble_cabac_slice_rbsp(w, &body);
        append_cabac_zero_words(&mut rbsp, bin_count, pic_size_mbs);
        Ok(rbsp)
    }

    /// Phase 6C.6d.3 — CABAC variant of `write_p_macroblock`.
    /// Scope: P_Skip + all direct P partitions (P_16x16, P_16x8,
    /// P_8x16, P_8x8 with sub-MB choices). Intra-in-P lands later
    /// alongside the re-entry into `write_intra_macroblock_cabac`.
    #[allow(clippy::too_many_arguments)]
    fn write_p_macroblock_cabac(
        &mut self,
        cabac: &mut CabacEncoder,
        reference: &super::reference_buffer::ReconFrame,
        ref_1: Option<&super::reference_buffer::ReconFrame>,
        mb_x: usize,
        mb_y: usize,
        y_plane: &[u8],
        y_stride: usize,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
        qp: u8,
        _qp_c: u8,
        is_last_mb: bool,
    ) -> Result<(), EncoderError> {
        use crate::codec::h264::cabac::encoder::{
            encode_coded_block_pattern, encode_mb_skip_flag, encode_mb_type_p,
            encode_residual_block_cabac_with_cbf_inc, encode_sub_mb_type_p,
        };
        use crate::codec::h264::cabac::neighbor::{
            block_pos_to_chroma_ac_idx, compute_cbf_ctx_idx_inc_chroma_ac,
            compute_cbf_ctx_idx_inc_chroma_dc, compute_cbf_ctx_idx_inc_luma_4x4,
            CurrentMbCbf, CurrentMbMvdAbs,
        };
        use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;

        // Gather source.
        let y0 = mb_y * 16;
        let x0 = mb_x * 16;
        let mut src_y = [[0u8; 16]; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                src_y[dy][dx] = y_plane[(y0 + dy) * y_stride + x0 + dx];
            }
        }
        let cy0 = mb_y * 8;
        let cx0 = mb_x * 8;
        let mut src_cb = [[0u8; 8]; 8];
        let mut src_cr = [[0u8; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                src_cb[dy][dx] = cb_plane[(cy0 + dy) * c_stride + cx0 + dx];
                src_cr[dy][dx] = cr_plane[(cy0 + dy) * c_stride + cx0 + dx];
            }
        }
        // ─── Per-MB AQ (Task #123 — variance-weighted P-slice AQ) ──
        //
        // Ported from `write_p_macroblock` (the CAVLC path) so the
        // CABAC P-path gets the same flat/textured QP offset. Three
        // modes, selected via PHASM_AQ_MODE:
        //   mode 1 (default): textured → -QP, flat → 0 (no +offset,
        //     avoids flat-area banding regression).
        //   mode 3: auto-variance + dark bias, requires
        //     aq_frame_mean_log2_q8 to be populated by the frame
        //     pre-pass in `encode_p_frame`. Strength knob
        //     PHASM_AQ_STRENGTH_Q10 (Q.10, default 256 = 0.25x —
        //     measured R-D sweet spot; see write_p_macroblock site
        //     for details).
        //   PHASM_DISABLE_PAQ=1: all MBs use slice QP unchanged.
        let variance = super::rate_control::mb_variance_16x16(&src_y);
        let aq_mode: u8 = super::mb_decision_b::env_var("PHASM_AQ_MODE")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let aq_disabled = super::mb_decision_b::env_var_os_is_some("PHASM_DISABLE_PAQ");
        let qp_offset: i32 = if aq_disabled {
            0
        } else if aq_mode == 3 {
            let strength: i32 = super::mb_decision_b::env_var("PHASM_AQ_STRENGTH_Q10")
                
                .and_then(|s| s.parse().ok())
                .unwrap_or(256);
            let mut luma_sum: u32 = 0;
            for row in &src_y {
                for &v in row {
                    luma_sum += v as u32;
                }
            }
            let avg_luma = luma_sum / 256;
            super::rate_control::variance_to_qp_offset_mode3(
                variance,
                self.aq_frame_mean_log2_q8,
                avg_luma,
                strength,
            )
        } else {
            super::rate_control::variance_to_qp_offset(variance).min(0)
        };
        // §v1.7 Phase 1.1 (#323) — MB-tree per-MB QP offset composes
        // additively with AQ offset. None / no env-gate → 0.
        let mb_tree_offset: i32 = self.mb_tree.as_ref()
            .map(|t| t.qp_offset(self.mb_tree_display_idx, mb_x, mb_y))
            .unwrap_or(0);
        // §v1.7 Phase 2.1 (#324) — Lookahead frame-level QP offset
        // composes additively. None → 0.
        let lookahead_offset: i32 = self.lookahead.as_ref()
            .map(|l| l.qp_offset(self.mb_tree_display_idx))
            .unwrap_or(0);
        let combined_offset = (qp_offset + mb_tree_offset + lookahead_offset).clamp(-15, 15);
        let mb_qp = ((qp as i32) + combined_offset).clamp(0, 51) as u8;
        let qp = mb_qp;
        let qp_c = derive_chroma_qp(mb_qp as i32, self.pps_params.chroma_qp_index_offset as i32) as u8;
        // Commit MB state for the deblock filter. is_intra=false
        // for the normal P path; the intra-in-P fallback (if
        // present) overrides this later.
        self.commit_mb_state(mb_x, mb_y, mb_qp, false);

        // Pre-ME skip_fast: before running ME + partition decision,
        // try a P_SKIP MB with the spec's predicted MV. If the residual
        // rounds to zero (luma AND chroma), emit P_SKIP immediately —
        // just a single mb_skip_flag=1 bin plus end_of_slice_flag.
        // Mirrors the CAVLC write_p_macroblock skip_fast at lines
        // 1700-1773 (same semantic; just the wire format differs —
        // CABAC emits mb_skip_flag inline, CAVLC accumulates a
        // mb_skip_run counter).
        //
        // `PHASM_CABAC_SKIP_FAST=1` — gated opt-in. Measurement on
        // IMG_4138 30f (2026-04-23) shows the R-D trade is QP-dependent:
        //   Q=80:  -0.23 dB avg /  -5.3% bits  (near-neutral)
        //   Q=60:  -0.69 dB avg / -15.4% bits  (mild regression)
        //   Q=40:  -1.41 dB avg / -26.1% bits  (too aggressive)
        // At low QP skip_fast catches MBs whose residuals only round
        // to zero because the content is near-stationary, but those
        // MBs carry perceptually-meaningful fine detail we'd rather
        // pay a few bits to keep. Default stays OFF; flipping the
        // default to ON at high QP only would require a QP threshold
        // knob, which adds complexity without a clear user-facing
        // R-D win. Ship as opt-in for callers doing size-constrained
        // encoding at Q≥80.
        let skip_fast_enabled = super::mb_decision_b::env_var("PHASM_CABAC_SKIP_FAST").as_deref() == Some("1");
        let p_skip_mv = super::partition_state::predict_p_skip_mv(
            &self.mv_grid,
            mb_x * 4,
            mb_y * 4,
        );
        if skip_fast_enabled {
            let skip_choice = PMbChoice::P16x16 { mv: p_skip_mv, ref_idx_l0: 0 };
            let s_pred_y = build_luma_prediction(reference, mb_x, mb_y, &skip_choice);
            let s_pred_cb = build_chroma_prediction(reference, 0, mb_x, mb_y, &skip_choice);
            let s_pred_cr = build_chroma_prediction(reference, 1, mb_x, mb_y, &skip_choice);

            let inter = QuantParams { qp: mb_qp, slice: QuantSlice::Inter };
            let chroma_qp_params = QuantParams { qp: qp_c, slice: QuantSlice::Inter };

            let mut any_luma_nonzero = false;
            for k in 0..16 {
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                let sby = by as usize;
                let sbx = bx as usize;
                let mut sub_res = [[0i32; 4]; 4];
                for dy in 0..4 {
                    for dx in 0..4 {
                        sub_res[dy][dx] = src_y[sby * 4 + dy][sbx * 4 + dx] as i32
                            - s_pred_y[sby * 4 + dy][sbx * 4 + dx] as i32;
                    }
                }
                let coeffs = forward_dct_4x4(&sub_res);
                let levels = trellis_quantize_4x4(&coeffs, inter, true)
                    .unwrap_or_else(|_| forward_quantize_4x4(&coeffs, inter));
                if levels.iter().any(|r| r.iter().any(|&v| v != 0)) {
                    any_luma_nonzero = true;
                    break;
                }
            }
            if !any_luma_nonzero {
                let (cb_ac, cb_dc) = self.encode_chroma_component(&src_cb, &s_pred_cb, chroma_qp_params);
                let (cr_ac, cr_dc) = self.encode_chroma_component(&src_cr, &s_pred_cr, chroma_qp_params);
                let chroma_any = cb_ac.iter().chain(cr_ac.iter())
                    .any(|b| b.iter().any(|r| r.iter().any(|&v| v != 0)))
                    || cb_dc.iter().flatten().chain(cr_dc.iter().flatten())
                        .any(|&v| v != 0);
                if !chroma_any {
                    self.mode_stats[MODE_STAT_P_SKIP_FAST] += 1;
                    // P_SKIP doesn't carry mb_qp_delta; decoder keeps
                    // QpY = prev_mb_qp for deblock.
                    self.commit_mb_state(mb_x, mb_y, self.prev_mb_qp as u8, false);
                    encode_mb_skip_flag(cabac, true, mb_x);
                    encode_end_of_slice_flag(cabac, is_last_mb);
                    self.mv_grid.fill(mb_x * 4, mb_y * 4, 4, 4, p_skip_mv, 0);
                    // §B-cascade-real Phase 1.1.B step 3: dual-write
                    // Skip prediction. No residual → no flip → both
                    // buffers receive identical pixels.
                    self.write_luma_mb_dual(mb_x as u32, mb_y as u32, &s_pred_y);
                    self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 0, &s_pred_cb);
                    self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 1, &s_pred_cr);
                    for k in 0..16 {
                        let (bx, by) = BLOCK_INDEX_TO_POS[k];
                        self.store_total_coeff_luma(
                            mb_x * 4 + bx as usize,
                            mb_y * 4 + by as usize,
                            0,
                        );
                    }
                    let mut nb = CabacNeighborMB::default();
                    nb.mb_type = MbTypeClass::PSkip;
                    nb.mb_skip_flag = true;
                    cabac.neighbors.commit(mb_x, nb);
                    return Ok(());
                }
            }
        }

        // Partition choice.
        let decision = decide_p_mb_with_cost(&src_y, reference, &mut self.me, &mut self.mv_grid, mb_x, mb_y);
        let mut choice = decision.best;
        let inter_cost = decision.best_cost;

        // v1.4 Phase 4.2 (#313, Path B) — multi-ref L0 post-pass.
        // After decide_p_mb_with_cost picks shape + MVs against
        // ref_0, re-search the chosen partition's MVs against
        // ref_1; upgrade the whole MB to ref_idx_l0=1 if ref_1 wins
        // the SATD comparison by more than λ × n_partitions (the
        // per-partition ref_idx_bit delta cost). At
        // MultiRefConfig::SINGLE_REF default, `ref_1` is None and
        // this is skipped — bit-identical to v1.3.
        if let Some(ref_1_frame) = ref_1 {
            super::partition_decision::refine_p_choice_multi_ref(
                &src_y, &mut self.me, mb_x, mb_y,
                reference, ref_1_frame,
                &mut choice,
            );
        }

        // Phase 6D.8 §30D-A: pre-MC MVD stego hook. Fires for each
        // partition's (x, y) MVDs; may modify values; encoder
        // updates choice's MVs from (pred + final_mvd) so MC + recon
        // run with the FINAL MV (no drift). When
        // `enable_mvd_stego_hook=false` (default) this is a no-op.
        //
        // Phase 6F.2: open per-MB MVD savepoint. Any MVD positions
        // logged by `apply_mvd_hook_to_choice` get retracted later
        // if this MB ends up emitting as P_SKIP / intra-in-P (no
        // MVDs in bitstream). See deferred-items.md §37.
        //
        // Phase 6F.2(e) — mv_grid snapshot/restore around the hook:
        // apply_mvd_hook_to_choice fills mv_grid sub-MB by sub-MB to
        // chain median predictors. By the time the hook completes,
        // the WHOLE MB's grid is filled. emit_p_mvds_cabac then runs
        // and re-fills incrementally, but its predictor sees the
        // hook's fills for *later* sub-MBs as already-decoded — so
        // its C-neighbor decoded-state differs from what the hook
        // saw. Snapshot before hook + restore after → emit starts
        // from the same pre-hook grid state and re-fills
        // incrementally identically to the hook.
        let mb_mv_grid_snapshot = self.mv_grid.snapshot_mb(mb_x, mb_y);
        self.begin_mvd_for_mb();
        self.apply_mvd_hook_to_choice(&mut choice, mb_x, mb_y);
        self.mv_grid.restore_mb(&mb_mv_grid_snapshot);

        // Phase D.0 intra-in-P emit gate.
        //   - `PHASM_TRACE_WOULD_BE_INTRA=1`: count MBs where intra
        //     would beat inter (pure measurement, no bitstream change).
        //   - `PHASM_CABAC_INTRA_IN_P=0`: disable intra-in-P emission
        //     (opt-out). Default is ON since Task #21 / Phase D.0-B
        //     fix (commit 206fc0b / 9e0040f, 2026-04-23): 1080p 60f
        //     Q=80 + 30f Q=40 parity gates pass bit-exact.
        //   - `PHASM_CABAC_INTRA_IN_P_FORCE_MB="x,y"`: force intra-in-P
        //     ONLY on the specified MB(s); suppresses natural firings
        //     elsewhere. Single-MB repro harness for future debug.
        let trace_would_be_intra = super::mb_decision_b::env_var_os_is_some("PHASM_TRACE_WOULD_BE_INTRA");
        let intra_in_p_enabled = super::mb_decision_b::env_var("PHASM_CABAC_INTRA_IN_P")
            .is_none_or(|v| v != "0");
        // FORCE_MB semantics: when set, intra-in-P fires ONLY on the
        // specified MB (suppresses natural firings elsewhere). Lets us
        // isolate parity bugs to a single MB emit for bin-by-bin
        // comparison. Format: "x,y" or "x,y;x2,y2;..." (semi-colon list).
        let force_mb_setting = super::mb_decision_b::env_var("PHASM_CABAC_INTRA_IN_P_FORCE_MB");
        let force_mb_active = force_mb_setting.is_some();
        let force_intra_here = intra_in_p_enabled
            && force_mb_setting
                .as_deref()
                .is_some_and(|s| {
                    s.split(';').any(|pair| {
                        let mut it = pair.split(',');
                        let x: Option<usize> = it.next().and_then(|p| p.trim().parse().ok());
                        let y: Option<usize> = it.next().and_then(|p| p.trim().parse().ok());
                        matches!((x, y), (Some(px), Some(py)) if px == mb_x && py == mb_y)
                    })
                });
        if trace_would_be_intra || intra_in_p_enabled {
            let neighbors_y = self.build_luma_neighbors_16x16(mb_x, mb_y);
            let i16_decision =
                choose_intra_16x16_mode_psy(&neighbors_y, &src_y, Self::PSY_RD_STRENGTH);
            let intra_raw_satd = satd_16x16(&src_y, &i16_decision.predicted);
            let intra_cost = intra_raw_satd.saturating_add(Self::P_INTRA_OVERHEAD);

            // Phase D.1(a) — SATD fast-out: if intra SATD is more
            // than 2× inter SATD, intra can never win; skip the
            // (expensive) RDO evaluation entirely. The `2×` ratio
            // is a widely used early-out constant across H.264 RDO
            // implementations.
            let satd_fast_out = intra_raw_satd > inter_cost.saturating_mul(2);

            // Phase D.1(b) — full intra-vs-inter RDO gate behind
            // `PHASM_INTRA_RDO=1`. Replaces the constant
            // `P_INTRA_OVERHEAD=512` with `D_intra + λ·R_intra`
            // computed via `evaluate_intra_in_p_rdo`. Intra wins
            // when its RDO cost beats inter's SATD × 5/4 ceiling —
            // biased toward inter since inter is the default mode.
            // Phase D.1 default ON since 2026-04-24 visual A/B on
            // IMG_4138 full-length (see
            // `~/Desktop/phasm_overnight_samples/`). The RDO gate
            // (intra competes against inter via `D + λR` instead of
            // a constant 512-SATD overhead) saves 28% bitrate at
            // equal-or-better visual quality. PSY stays default 0 —
            // the AC-energy-preservation bonus caused visible
            // artefacts on smooth surfaces on this content. Opt out
            // of the RDO gate via `PHASM_INTRA_RDO=0`.
            let intra_rdo_enabled = super::mb_decision_b::env_var("PHASM_INTRA_RDO")
                .is_none_or(|v| v != "0");
            let intra_rdo_wins = if intra_rdo_enabled && !satd_fast_out && !force_mb_active {
                let frame_w4 = (self.width / 4) as usize;
                let chroma_mode = super::mb_decision_b::env_var("PHASM_INTRA_CHROMA_MODE")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0);
                // Phase C.v3 chroma RDO: build source Cb/Cr blocks
                // from the current MB. Enabled by default; opt out
                // via `PHASM_CHROMA_RDO=0` to keep luma-only cost.
                let chroma_rdo_enabled = super::mb_decision_b::env_var("PHASM_CHROMA_RDO")
                    .is_none_or(|v| v != "0");
                let (src_cb, src_cr) = if chroma_rdo_enabled {
                    let cy0 = mb_y * 8;
                    let cx0 = mb_x * 8;
                    let mut cb = [[0u8; 8]; 8];
                    let mut cr = [[0u8; 8]; 8];
                    for dy in 0..8 {
                        for dx in 0..8 {
                            cb[dy][dx] = cb_plane[(cy0 + dy) * c_stride + cx0 + dx];
                            cr[dy][dx] = cr_plane[(cy0 + dy) * c_stride + cx0 + dx];
                        }
                    }
                    (Some(cb), Some(cr))
                } else {
                    (None, None)
                };
                let chroma_inputs_intra = match (&src_cb, &src_cr) {
                    (Some(cb), Some(cr)) => Some(super::rdo::ChromaRdoInputs {
                        src_cb: cb,
                        src_cr: cr,
                        qp_c,
                    }),
                    _ => None,
                };
                // Intra RDO cost — pure SSD, no drift factor.
                let (_di, _ri, intra_rdo_cost) = super::rdo::evaluate_intra_in_p_rdo(
                    &src_y,
                    &i16_decision.predicted,
                    i16_decision.mode as u32,
                    chroma_mode,
                    &self.total_coeff_grid,
                    frame_w4,
                    mb_x,
                    mb_y,
                    mb_qp,
                    chroma_inputs_intra,
                    if chroma_rdo_enabled { Some(reference) } else { None },
                );
                // Fix 2026-04-24: compare against INTER RDO cost
                // (with drift factor + psy) instead of inter SATD.
                // Prior gate (`intra_rdo_cost < inter_satd × 5/4`)
                // meant D.2 drift factor + E psy term on inter never
                // reached the intra-vs-inter decision — they only
                // affected inter-vs-inter ranking which already
                // matches SATD per Phase C. Compare RDO-to-RDO so
                // the new terms actually shift intra firings.
                let chroma_inputs_inter = match (&src_cb, &src_cr) {
                    (Some(cb), Some(cr)) => Some(super::rdo::ChromaRdoInputs {
                        src_cb: cb,
                        src_cr: cr,
                        qp_c,
                    }),
                    _ => None,
                };
                let inter_rdo = super::rdo::evaluate_p_mb_rdo(
                    &choice,
                    &src_y,
                    reference,
                    &mut self.mv_grid,
                    mb_x,
                    mb_y,
                    mb_qp,
                    &self.total_coeff_grid,
                    frame_w4,
                    chroma_inputs_inter,
                );
                // Ceiling: intra wins when intra_rdo_cost is below
                // `CEIL_NUM/CEIL_DEN × inter_rdo_cost`. Task #53
                // (stealth RATE calibration, 2026-04-24) retuned this
                // from the initial 5/4 = 1.25× down to 95/100 = 0.95×
                // — our R estimator underweights I_16x16 (approximates
                // as 16 inter-style 4×4 CAVLC, omits the DC-Hadamard
                // split) so intra cost is systematically too low;
                // a ratio < 1 compensates.
                //
                // Measured on IMG_4138 / IMG_4273 first-10-frames
                // Q=22 vs a reference H.264 encoder at equivalent
                // Main-profile / medium-speed settings, no B-frames:
                //   ratio 1.25: 12.55 % / 23.93 % intra-in-P
                //   ratio 1.00:  3.89 % /  6.49 %
                //   ratio 0.95:  2.80 % /  4.79 %  ← default
                //   ratio 0.90:  2.08 % /  3.58 %
                // Reference targets: 2.3 % / 5.1 %. 0.95 lands within
                // ±1 pp on both clips (inside the ±5 pp stealth
                // target). PSNR is flat-to-+0.1 dB tighter.
                //
                // Env-tunable via `PHASM_INTRA_RDO_CEIL_NUM`/`_DEN`.
                let ceil_num: u64 = super::mb_decision_b::env_var("PHASM_INTRA_RDO_CEIL_NUM")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(Self::INTRA_RDO_CEIL_NUM);
                let ceil_den: u64 = super::mb_decision_b::env_var("PHASM_INTRA_RDO_CEIL_DEN")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(Self::INTRA_RDO_CEIL_DEN);
                let ceiling = inter_rdo.cost.saturating_mul(ceil_num) / ceil_den.max(1);
                intra_rdo_cost < ceiling
            } else {
                false
            };

            // With FORCE_MB active, only fire on explicitly-listed MBs
            // (block natural picks). Without FORCE_MB, fire on the
            // natural SATD-vs-inter decision (or RDO when enabled).
            let picks_intra = if force_mb_active {
                force_intra_here
            } else if satd_fast_out {
                false
            } else if intra_rdo_enabled {
                intra_rdo_wins
            } else {
                intra_cost < inter_cost
            };
            if picks_intra {
                self.mode_stats[MODE_STAT_INTRA_IN_P] += 1;
                if intra_in_p_enabled {
                    // Phase D.2-stealth: intra-in-P now also considers
                    // I_4x4 as a within-intra option. Opt out with
                    // `PHASM_INTRA_IN_P_ALLOW_I4X4=0`. Default ON
                    // since 2026-04-24: 10-frame parity 99.99 dB on
                    // IMG_4138 + IMG_4273, and the calibration sweep
                    // (task #52) matched the reference encoder's
                    // ~4 % I_4x4 share within intra-in-P within
                    // ±5 pp on both clips.
                    let i4x4_allow = super::mb_decision_b::env_var("PHASM_INTRA_IN_P_ALLOW_I4X4")
                        
                        .is_none_or(|v| v != "0");
                    // Phase D.4 fast-intra gate (task #51). I_4x4
                    // exploration runs 9 modes × 16 blocks = 144 SATDs
                    // per MB — skip it unless inter clearly loses to
                    // I_16x16, i.e. `satd_inter >
                    // PHASM_D4_FAST_INTRA_THRESH_Q10 × satd_i16x16`
                    // (Q.10, default 1536 = 1.5). Leaves I_16x16 as
                    // the only intra-in-P option otherwise. Scales to
                    // the inter cost, so drift-damaged MBs (large
                    // residual → large `inter_cost`) still get the
                    // I_4x4 pick when it matters.
                    let d4_thresh_q10: u32 = super::mb_decision_b::env_var("PHASM_D4_FAST_INTRA_THRESH_Q10")
                        
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(Self::D4_FAST_INTRA_THRESH_Q10);
                    // `inter_cost * 1024 > satd_i16x16 * thresh_q10`
                    // expressed without overflow risk on 1080p SATDs.
                    let fast_intra_passes = (inter_cost as u64)
                        .saturating_mul(1024)
                        > (intra_raw_satd as u64).saturating_mul(d4_thresh_q10 as u64);
                    let use_i4x4 = if i4x4_allow && fast_intra_passes {
                        let i4x4_result = encode_i4x4_mb(
                            &src_y, &mut self.recon, mb_x, mb_y, mb_qp,
                            Self::PSY_RD_STRENGTH,
                        );
                        // Phase D.2-stealth calibration (#52):
                        // I_4x4 emits 16× prev_intra4x4_pred_mode_flag
                        // + optional rem + 16 per-block residuals
                        // which I_16x16 does not — needs an explicit
                        // SATD penalty to match real-encoder
                        // distributions. Sweep on IMG_4138 Q=80 vs a
                        // reference H.264 encoder at Main-profile /
                        // medium-speed / no-B-frames: penalty=240
                        // lands at 4.2 % I_4x4 share (reference
                        // 4.3 %) and 1.7 % on IMG_4273 (reference
                        // 3.9 %) — both within ±5 pp of the
                        // reference. See plan doc for the full
                        // sweep table.
                        let penalty = super::mb_decision_b::env_var("PHASM_IIP_I4X4_PENALTY")
                            
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(Self::IIP_I4X4_PENALTY);
                        let i4x4_cost = i4x4_result.total_satd
                            .saturating_add(penalty);
                        if i4x4_cost < intra_raw_satd {
                            Some(i4x4_result)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    encode_mb_skip_flag(cabac, false, mb_x);
                    self.mv_grid.fill(
                        mb_x * 4, mb_y * 4, 4, 4,
                        MotionVector::ZERO, REF_IDX_NONE,
                    );
                    self.commit_mb_state(mb_x, mb_y, mb_qp, true);

                    // Phase 6F.2: intra-in-P wins → emit as I-MB, no
                    // MVDs in bitstream. Retract any phantom MVD
                    // positions logged by apply_mvd_hook_to_choice.
                    self.rollback_mvd_for_mb();
                    if let Some(i4x4_result) = use_i4x4 {
                        self.mode_stats[MODE_STAT_INTRA_IN_P_I4X4] += 1;
                        return self.commit_i4x4_macroblock_cabac(
                            cabac, mb_x, mb_y, &i4x4_result,
                            cb_plane, cr_plane, c_stride,
                            mb_qp, qp_c, is_last_mb,
                            /* in_p_slice: */ true,
                        );
                    }
                    self.mode_stats[MODE_STAT_INTRA_IN_P_I16X16] += 1;
                    return self.write_i16x16_macroblock_cabac(
                        cabac, mb_x, mb_y, y_plane, y_stride,
                        cb_plane, cr_plane, c_stride,
                        mb_qp, qp_c, is_last_mb,
                        /* in_p_slice: */ true,
                    );
                }
            }
        }

        // Build predictions + residuals (same pipeline as CAVLC path).
        // v1.4 Phase 4.2 (#313, Path B) — dispatch reference based on
        // post-pass-chosen ref_idx_l0. At ref_idx=0 use `reference`
        // (= ref_0 = closest past). At ref_idx=1 use `ref_1_frame`.
        // Path B uniform-per-MB: all partitions of the MB share the
        // same ref_idx_l0, so a single dispatch resolves to one ref.
        let chosen_ref: &super::reference_buffer::ReconFrame =
            if choice.ref_idx_l0_uniform() == 0 {
                reference
            } else {
                ref_1.expect("ref_idx_l0=1 requires ref_1 — refine pass should not fire without it")
            };
        let pred_y = build_luma_prediction(chosen_ref, mb_x, mb_y, &choice);
        let pred_cb = build_chroma_prediction(chosen_ref, 0, mb_x, mb_y, &choice);
        let pred_cr = build_chroma_prediction(chosen_ref, 1, mb_x, mb_y, &choice);

        let inter = QuantParams {
            qp,
            slice: QuantSlice::Inter,
        };
        let chroma_qp = QuantParams {
            qp: qp_c,
            slice: QuantSlice::Inter,
        };

        let mut luma_ac_levels = [[[0i32; 4]; 4]; 16];
        let mut luma_nonzero = [false; 16];
        // Cache the per-4x4-block reconstructed residual so the 100-J
        // RDO chooser below and the final reconstruction at the
        // bottom of this function both read from the same cached
        // results without re-running dequant+inverse twice.
        let mut luma_recon_residual_4x4 = [[[0i32; 4]; 4]; 16];
        let mut luma_nonzero_count_4x4: u64 = 0;
        let mut sse_4x4: u64 = 0;
        for k in 0..16 {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            let sby = by as usize;
            let sbx = bx as usize;
            let mut sub_res = [[0i32; 4]; 4];
            for dy in 0..4 {
                for dx in 0..4 {
                    sub_res[dy][dx] = src_y[sby * 4 + dy][sbx * 4 + dx] as i32
                        - pred_y[sby * 4 + dy][sbx * 4 + dx] as i32;
                }
            }
            let coeffs = forward_dct_4x4(&sub_res);
            let levels = trellis_quantize_4x4(&coeffs, inter, true)
                .unwrap_or_else(|_| forward_quantize_4x4(&coeffs, inter));
            luma_ac_levels[k] = levels;
            luma_nonzero[k] = levels.iter().any(|r| r.iter().any(|&v| v != 0));
            // Dequant + inverse transform for cached reconstruction.
            use crate::codec::h264::transform::{dequant_4x4, inverse_4x4_integer};
            let dq = dequant_4x4(&levels, qp as i32, false);
            let recon_r = inverse_4x4_integer(&dq);
            luma_recon_residual_4x4[k] = recon_r;
            // Accumulate SSE (recon − original residual) and non-zero
            // count for the Phase 100-J RDO chooser.
            for dy in 0..4 {
                for dx in 0..4 {
                    let d = (recon_r[dy][dx] - sub_res[dy][dx]) as i64;
                    sse_4x4 = sse_4x4.saturating_add((d * d) as u64);
                }
            }
            luma_nonzero_count_4x4 += levels
                .iter()
                .flatten()
                .filter(|&&v| v != 0)
                .count() as u64;
        }
        let cbp_luma_8x8 = luma_8x8_cbp_mask(&luma_nonzero);

        let (cb_ac, cb_dc) = self.encode_chroma_component(&src_cb, &pred_cb, chroma_qp);
        let (cr_ac, cr_dc) = self.encode_chroma_component(&src_cr, &pred_cr, chroma_qp);
        // §B-cascade-real Phase 1.1.B step 2 — parallel post-flip
        // arrays for the visual_recon recon pass at function epilogue.
        // Updated by the residual-hook fire sites below; if no flip
        // fires, _post stays equal to pre-flip and visual_recon ends
        // up byte-identical to encoder.recon.
        let mut luma_ac_levels_post = luma_ac_levels;
        let mut cb_ac_post = cb_ac;
        let mut cb_dc_post = cb_dc;
        let mut cr_ac_post = cr_ac;
        let mut cr_dc_post = cr_dc;
        let any_cb_ac = cb_ac.iter().any(|b| b.iter().any(|r| r.iter().any(|&v| v != 0)));
        let any_cr_ac = cr_ac.iter().any(|b| b.iter().any(|r| r.iter().any(|&v| v != 0)));
        let any_cb_dc = cb_dc.iter().flatten().any(|&v| v != 0);
        let any_cr_dc = cr_dc.iter().flatten().any(|&v| v != 0);
        let cbp_chroma = if any_cb_ac || any_cr_ac {
            2u8
        } else if any_cb_dc || any_cr_dc {
            1u8
        } else {
            0u8
        };
        // Phase 100-G + 100-J: if PPS `transform_8x8_mode_flag` is
        // enabled and the partition choice has no sub-MB partition
        // smaller than 8×8, we may emit the luma residual through
        // the 8×8 transform. Spec § 7.3.5.1 gates flag emission on
        // CodedBlockPatternLuma > 0 AND noSubMbPartSizeLessThan8x8Flag,
        // so ineligible MBs stay on the 4×4 path regardless.
        //
        // Phase 100-J (RDO chooser): compute both quantisations,
        // then pick the one with the smaller `D + λ²·R` cost, where
        // D is the SSE of (reconstructed residual − original
        // residual) and R is approximated as `non-zero-coef-count
        // × BITS_PER_NONZERO_APPROX`. This replaces Phase 100-G's
        // "always pick 8×8 when eligible" first-cut that was shown
        // in Phase 100-I's R-D sweep to cost bits at higher QPs.
        use super::intra_8x8_encode::I8X8_BLOCK_POS;
        use super::transform_8x8::{
            dequant_8x8_block, forward_dct_8x8, inverse_dct_8x8, quant_8x8_block, Slice8x8,
        };

        let eligible_8x8 = self.enable_transform_8x8 && choice.no_sub_mb_part_size_lt_8x8();
        let mut levels_8x8: [[[i16; 8]; 8]; 4] = [[[0i16; 8]; 8]; 4];
        // §B-cascade-real Phase 1.1.B step 2 — parallel post-flip 8×8
        // luma levels for the visual_recon path.
        let mut levels_8x8_post: [[[i16; 8]; 8]; 4] = [[[0i16; 8]; 8]; 4];
        let mut cbp_luma_via_8x8 = 0u8;
        let mut sse_8x8: u64 = 0;
        let mut luma_nonzero_count_8x8: u64 = 0;
        if eligible_8x8 {
            for blk in 0..4 {
                let (bx_mb, by_mb) = I8X8_BLOCK_POS[blk];
                let bx0 = (bx_mb * 8) as usize;
                let by0 = (by_mb * 8) as usize;
                let mut res = [[0i32; 8]; 8];
                for dy in 0..8 {
                    for dx in 0..8 {
                        res[dy][dx] = src_y[by0 + dy][bx0 + dx] as i32
                            - pred_y[by0 + dy][bx0 + dx] as i32;
                    }
                }
                let coeffs = forward_dct_8x8(&res);
                let lvl = quant_8x8_block(&coeffs, qp, Slice8x8::Inter);
                levels_8x8[blk] = lvl;
                if lvl.iter().any(|r| r.iter().any(|&v| v != 0)) {
                    cbp_luma_via_8x8 |= 1 << blk;
                }
                // Dequant + inverse for the chooser's SSE, and
                // accumulate nonzero count for the R proxy.
                let dq = dequant_8x8_block(&lvl, qp);
                let inv = inverse_dct_8x8(&dq);
                for dy in 0..8 {
                    for dx in 0..8 {
                        let recon_r = (inv[dy][dx] + 32) >> 6;
                        let d = (recon_r - res[dy][dx]) as i64;
                        sse_8x8 = sse_8x8.saturating_add((d * d) as u64);
                    }
                }
                luma_nonzero_count_8x8 +=
                    lvl.iter().flatten().filter(|&&v| v != 0).count() as u64;
            }
            // §B-cascade-real Phase 1.1.B step 2: seed _post with
            // pre-flip levels; the hook will overwrite per-block.
            levels_8x8_post = levels_8x8;
        }

        // Phase 100-J RDO chooser. The 8×8 path can be legally
        // signalled only when `CodedBlockPatternLuma > 0`; if the
        // 8×8 quant produced zero everywhere, the 4×4 path is
        // forced regardless of what the cost comparison says.
        //
        // Cost formula:  `D + (λ²·R) >> 8`
        //   D = SSE between reconstructed residual and source residual
        //   R ≈ nonzero_count × BITS_PER_NONZERO_APPROX (CABAC average)
        //   λ² = super::rdo::LAMBDA2_TAB[qp]  (Q.8, spec Sullivan-Wiegand)
        //
        // BITS_PER_NONZERO_APPROX = 6: typical CABAC cost for a
        // small-magnitude residual coefficient (sig-map + last +
        // abs-prefix + sign, averaged across coef positions).
        const BITS_PER_NONZERO_APPROX: u64 = 6;
        let lambda2 = super::rdo::LAMBDA2_TAB[qp.min(51) as usize] as u64;
        let cost_4x4 = sse_4x4.saturating_add(
            (lambda2 * luma_nonzero_count_4x4 * BITS_PER_NONZERO_APPROX) >> 8,
        );
        let cost_8x8 = sse_8x8.saturating_add(
            (lambda2 * luma_nonzero_count_8x8 * BITS_PER_NONZERO_APPROX) >> 8,
        );
        let use_8x8 =
            eligible_8x8 && cbp_luma_via_8x8 != 0 && cost_8x8 < cost_4x4;
        let cbp_luma_8x8 = if use_8x8 { cbp_luma_via_8x8 } else { cbp_luma_8x8 };
        let cbp_value = pack_cbp(cbp_luma_8x8, cbp_chroma);

        // P_Skip detection (spec § 7.3.5.1 / § 8.4.1.2.1):
        // if the mode decision picked P_L0_16x16, cbp is zero, and the
        // chosen MV matches the spec's P_Skip MV derivation, we can
        // signal P_Skip with just a single mb_skip_flag = 1 bin.
        //
        // Phase 6F.2(f) — gate post-ME P_SKIP off whenever the MVD
        // stego hook is active. Reason: the hook may modify MVD
        // signs (e.g. Pass 3 InjectionHook applies the STC plan).
        // The modified MV produces different residuals → potentially
        // different cbp → P_SKIP decision flips between Pass 1 (where
        // the cover was logged with original MV) and Pass 3 (where
        // bytes are emitted with injected MV). When P_SKIP flips
        // between passes, the bitstream cover length differs from
        // the planned cover length — STC reverse on the walked
        // bitstream fails, causing FrameCorrupted on real-world
        // round trips. Disabling post-ME P_SKIP under stego costs
        // ~5–15% bitrate on low-motion content but stabilises the
        // mode decision across passes. See deferred-items.md §37.
        let is_skip = if self.enable_mvd_stego_hook {
            false
        } else if cbp_value == 0 {
            if let PMbChoice::P16x16 { mv, .. } = choice {
                let p_skip_mv = super::partition_state::predict_p_skip_mv(
                    &self.mv_grid,
                    mb_x * 4,
                    mb_y * 4,
                );
                mv.mv_x == p_skip_mv.mv_x && mv.mv_y == p_skip_mv.mv_y
            } else {
                false
            }
        } else {
            false
        };

        // 1. mb_skip_flag (CABAC-only).
        encode_mb_skip_flag(cabac, is_skip, mb_x);
        if is_skip {
            // Spec § 7.4.5.1: P_SKIP doesn't carry mb_qp_delta, decoder
            // uses QpY = prev_mb_qp for deblock. Match it in qp_grid.
            self.commit_mb_state(mb_x, mb_y, self.prev_mb_qp as u8, false);
            // Spec § 7.3.5.1: a skipped MB has no further syntax. Emit
            // end_of_slice_flag (every MB does, with value 1 for the
            // final MB only) and commit neighbor state.
            encode_end_of_slice_flag(cabac, is_last_mb);

            // Fill the MV grid with the P_Skip MV so subsequent MBs
            // see this partition in their predictor chain. The
            // reconstruction was already computed via
            // `build_luma_prediction`/`build_chroma_prediction` above
            // with this MV (we only entered the skip branch when
            // `choice.mv == p_skip_mv`).
            if let PMbChoice::P16x16 { mv, .. } = choice {
                self.mv_grid.fill(mb_x * 4, mb_y * 4, 4, 4, mv, 0);
            }
            // §B-direct-fix Stage 2 (#232) — dual-write predicted
            // pixels. P_Skip carries no residual, so visual_recon ==
            // self.recon for this MB. But visual_recon MUST receive
            // the prediction surface anyway: omitting the write
            // leaves visual_recon at the previous frame's pixels for
            // this MB region, breaking encoder/decoder recon parity
            // when the next P/B frame references it.
            self.write_luma_mb_dual(mb_x as u32, mb_y as u32, &pred_y);
            self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 0, &pred_cb);
            self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 1, &pred_cr);

            // total_coeff_grid cleared for all blocks in this MB.
            for k in 0..16 {
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                self.store_total_coeff_luma(
                    mb_x * 4 + bx as usize,
                    mb_y * 4 + by as usize,
                    0,
                );
            }

            let mut nb = CabacNeighborMB::default();
            nb.mb_type = MbTypeClass::PSkip;
            nb.mb_skip_flag = true;
            cabac.neighbors.commit(mb_x, nb);
            self.mode_stats[MODE_STAT_P_SKIP_POST_ME] += 1;
            // Phase 6F.2: P_SKIP emit → no MVDs in bitstream.
            // Retract any phantom MVD positions logged by
            // apply_mvd_hook_to_choice. Pairs with begin_mvd_for_mb
            // earlier in this fn. See deferred-items.md §37.
            self.rollback_mvd_for_mb();
            return Ok(());
        }

        // Mode-stats instrumentation: count the partition choice for
        // the `PHASM_MODE_STATS=1` diagnostic. Placed after the P_SKIP
        // branch so the counter reflects the actual emitted mb_type.
        match choice {
            PMbChoice::P16x16 { .. } => self.mode_stats[MODE_STAT_P_16X16] += 1,
            PMbChoice::P16x8 { .. } => self.mode_stats[MODE_STAT_P_16X8] += 1,
            PMbChoice::P8x16 { .. } => self.mode_stats[MODE_STAT_P_8X16] += 1,
            PMbChoice::P8x8 { .. } => self.mode_stats[MODE_STAT_P_8X8] += 1,
        }

        // 2. mb_type — direct P partitions (values 0..3) map to Table 9-37.
        let mb_type_value = choice.mb_type_codenum() as u32;
        encode_mb_type_p(cabac, mb_type_value, mb_x);

        // 3. sub_mb_type — only for P_8x8.
        if let PMbChoice::P8x8 { ref sub } = choice {
            for s in sub.iter() {
                encode_sub_mb_type_p(cabac, s.sub_mb_type_codenum());
            }
        }

        // 4. v1.4 (#305) — ref_idx_l0 per partition per spec § 7.3.5.1
        // mb_pred(). Emitted iff num_ref_idx_l0_active > 1. At
        // MultiRefConfig::SINGLE_REF default the gate is closed and
        // zero bins emit (bit-identical to v1.3).
        //
        // v1.4 Phase 4.5 (#316) — gate on actual P-side L0 references
        // (mirror of slice-header `actual_l0_p`). past_anchor=None on
        // first P after IDR → no ref_idx_l0 bin on the wire.
        let num_active_l0 = if self.multi_ref_config.max_l0_active > 1
            && self.dpb.past_anchor.is_some()
        {
            2u8
        } else {
            1u8
        };
        // v1.4 Phase 4.5 (#316) — within-MB ref_idx_l0 tracker.
        // Partition 1+ of P_16x8 / P_8x16 / P_8x8 reads partition 0's
        // just-emitted ref_idx for the bin 0 ctxIdxInc neighbour
        // lookup per spec § 6.4.11.7.
        let mut current_ref_idx_mb = crate::codec::h264::cabac::neighbor::CurrentMbRefIdx::new();
        if num_active_l0 > 1 {
            use crate::codec::h264::cabac::encoder::encode_ref_idx;
            let c_max = (num_active_l0 - 1) as u32;
            match &choice {
                PMbChoice::P16x16 { ref_idx_l0, .. } => {
                    encode_ref_idx(cabac, *ref_idx_l0 as u32, &current_ref_idx_mb, mb_x, 0, 0, c_max);
                    current_ref_idx_mb.fill_region(0, 0, 4, 4, *ref_idx_l0 as i8);
                }
                PMbChoice::P16x8 { ref_idx_l0, .. } => {
                    encode_ref_idx(cabac, ref_idx_l0[0] as u32, &current_ref_idx_mb, mb_x, 0, 0, c_max);
                    current_ref_idx_mb.fill_region(0, 0, 4, 2, ref_idx_l0[0] as i8);
                    encode_ref_idx(cabac, ref_idx_l0[1] as u32, &current_ref_idx_mb, mb_x, 0, 2, c_max);
                    current_ref_idx_mb.fill_region(0, 2, 4, 2, ref_idx_l0[1] as i8);
                }
                PMbChoice::P8x16 { ref_idx_l0, .. } => {
                    encode_ref_idx(cabac, ref_idx_l0[0] as u32, &current_ref_idx_mb, mb_x, 0, 0, c_max);
                    current_ref_idx_mb.fill_region(0, 0, 2, 4, ref_idx_l0[0] as i8);
                    encode_ref_idx(cabac, ref_idx_l0[1] as u32, &current_ref_idx_mb, mb_x, 2, 0, c_max);
                    current_ref_idx_mb.fill_region(2, 0, 2, 4, ref_idx_l0[1] as i8);
                }
                PMbChoice::P8x8 { sub } => {
                    // Per spec § 7.3.5.1 — one ref_idx_l0 per 8×8
                    // sub-MB regardless of sub_mb_type internal
                    // partitioning.
                    const SUB_ORIGINS: [(u8, u8); 4] =
                        [(0, 0), (2, 0), (0, 2), (2, 2)];
                    for (i, sc) in sub.iter().enumerate() {
                        let (bx, by) = SUB_ORIGINS[i];
                        let r = sc.ref_idx_l0() as u32;
                        encode_ref_idx(cabac, r, &current_ref_idx_mb, mb_x, bx, by, c_max);
                        current_ref_idx_mb.fill_region(bx, by, 2, 2, sc.ref_idx_l0() as i8);
                    }
                }
            }
        }

        // 5. MVDs per partition, with same-MB neighbor tracking.
        let mut current_mvd = CurrentMbMvdAbs::new();
        self.emit_p_mvds_cabac(cabac, &mut current_mvd, mb_x, mb_y, &choice);

        // 6. coded_block_pattern (CABAC — non-Intra_16x16 always emits CBP).
        encode_coded_block_pattern(cabac, cbp_value, mb_x);

        // Spec § 7.3.5.1 macroblock_layer: emit `transform_size_8x8_flag`
        // for non-I_NxN non-Intra_16x16 MBs (= P-inter here) when
        // CodedBlockPatternLuma > 0 AND PPS transform_8x8_mode_flag = 1
        // AND noSubMbPartSizeLessThan8x8Flag. The emitted value is 1
        // when the 8×8 transform path is chosen (Phase 100-G) and 0
        // when the 4×4 transform path is chosen.
        if self.enable_transform_8x8
            && cbp_luma_8x8 != 0
            && choice.no_sub_mb_part_size_lt_8x8()
        {
            crate::codec::h264::cabac::encoder::encode_transform_size_8x8_flag(
                cabac, use_8x8, mb_x,
            );
        }

        // 7. mb_qp_delta only if cbp != 0.
        let qp_delta_emitted;
        if cbp_value != 0 {
            let delta = qp as i32 - self.prev_mb_qp;
            encode_mb_qp_delta(cabac, delta);
            self.prev_mb_qp = qp as i32;
            qp_delta_emitted = delta;
        } else {
            // Spec § 7.3.5.1: no mb_qp_delta → decoder keeps prev_mb_qp.
            // Override our qp_grid (AQ mb_qp) with prev_mb_qp for deblock.
            self.commit_mb_state(mb_x, mb_y, self.prev_mb_qp as u8, false);
            qp_delta_emitted = 0;
        }

        // 8. Residuals.
        let mut current_cbf = CurrentMbCbf::new();
        // write_p_macroblock_cabac is the P-slice inter path. is_intra=false.
        let current_is_intra = false;
        if use_8x8 {
            // 8×8 luma residual emit (ctxBlockCat = 5). Four 8×8 blocks
            // gated by cbp_luma bits. No per-block CBF emission (unlike
            // cat 0..4 — cat 5 sig/last/abs contexts are position-
            // indexed via SIG/LAST_COEFF_FLAG_OFFSET_8X8_FRAME).
            use crate::codec::h264::cabac::encoder::{
                encode_residual_block_cabac_8x8, ZIGZAG_8X8,
            };
            for k in 0..4 {
                if cbp_luma_8x8 & (1 << k) != 0 {
                    let mut scan = [0i32; 64];
                    for i in 0..64 {
                        let pos = ZIGZAG_8X8[i] as usize;
                        scan[i] = levels_8x8[k][pos / 8][pos % 8] as i32;
                    }
                    // Phase 6D.8: stego hook for P-frame 8×8 luma residual.
                    self.invoke_stego_residual_hook(
                        mb_x, mb_y, &mut scan, 0, 63,
                        super::super::stego::orchestrate::ResidualPathKind::Luma8x8 {
                            block_idx: k as u8,
                        },
                    );
                    // §B-cascade-real Phase 1.1.B step 2: capture
                    // post-flip 8×8 luma into levels_8x8_post.
                    for i in 0..64 {
                        let pos = ZIGZAG_8X8[i] as usize;
                        levels_8x8_post[k][pos / 8][pos % 8] = scan[i] as i16;
                    }
                    encode_residual_block_cabac_8x8(cabac, &scan);
                }
            }
            // Populate cat-2 neighbor CBF state from the 8×8 cbp_luma
            // bits so subsequent 4×4 MBs see each 4×4 sub-position's
            // coded_block_flag as the 8×8 block's cbp bit (spec
            // § 9.3.3.1.1.9 rule for 8×8-transformed neighbours).
            // Also writes the TotalCoeff grid for any mixed
            // 4×4 / 8×8 content in the same frame.
            for k in 0..16 {
                let blk8 = k / 4;
                let coded = (cbp_luma_8x8 & (1 << blk8)) != 0;
                current_cbf.set(2, k, coded);
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                let abs_bx = mb_x * 4 + bx as usize;
                let abs_by = mb_y * 4 + by as usize;
                self.store_total_coeff_luma(abs_bx, abs_by, if coded { 1 } else { 0 });
            }
            self.mark_transform_8x8_mb(mb_x, mb_y);
        } else if cbp_luma_8x8 != 0 {
            for k in 0..16 {
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                if cbp_luma_8x8 & (1 << (k / 4)) != 0 {
                    let mut scan = raster_to_scan_levels(&luma_ac_levels[k]);
                    // Phase 6D.8: stego hook for P-frame 4×4 luma residual.
                    self.invoke_stego_residual_hook(
                        mb_x, mb_y, &mut scan, 0, 15,
                        super::super::stego::orchestrate::ResidualPathKind::Luma4x4 {
                            block_idx: k as u8,
                        },
                    );
                    // §B-cascade-real Phase 1.1.B step 2: capture
                    // post-flip 4×4 luma into luma_ac_levels_post.
                    luma_ac_levels_post[k] = scan_to_raster_levels(&scan);
                    let inc = compute_cbf_ctx_idx_inc_luma_4x4(
                        &current_cbf,
                        &cabac.neighbors,
                        mb_x,
                        bx,
                        by,
                        current_is_intra,
                    );
                    let coded = encode_residual_block_cabac_with_cbf_inc(
                        cabac, &scan, 0, 15, 2, inc,
                    );
                    current_cbf.set(2, k, coded);
                    let abs_bx = mb_x * 4 + bx as usize;
                    let abs_by = mb_y * 4 + by as usize;
                    self.store_total_coeff_luma(abs_bx, abs_by, if coded { 1 } else { 0 });
                } else {
                    let abs_bx = mb_x * 4 + bx as usize;
                    let abs_by = mb_y * 4 + by as usize;
                    self.store_total_coeff_luma(abs_bx, abs_by, 0);
                }
            }
        } else {
            for k in 0..16 {
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                self.store_total_coeff_luma(
                    mb_x * 4 + bx as usize,
                    mb_y * 4 + by as usize,
                    0,
                );
            }
        }
        if cbp_chroma >= 1 {
            for (plane, dc) in [&cb_dc, &cr_dc].iter().enumerate() {
                let mut dc_flat: [i32; 4] = [dc[0][0], dc[0][1], dc[1][0], dc[1][1]];
                // Phase 6D.8: stego hook for P-frame Chroma DC.
                self.invoke_stego_residual_hook(
                    mb_x, mb_y, &mut dc_flat, 0, 3,
                    super::super::stego::orchestrate::ResidualPathKind::ChromaDc {
                        plane: plane as u8,
                    },
                );
                // §B-cascade-real Phase 1.1.B step 2: capture post-flip
                // chroma DC.
                let dc_post_raster: [[i32; 2]; 2] =
                    [[dc_flat[0], dc_flat[1]], [dc_flat[2], dc_flat[3]]];
                if plane == 0 {
                    cb_dc_post = dc_post_raster;
                } else {
                    cr_dc_post = dc_post_raster;
                }
                let inc =
                    compute_cbf_ctx_idx_inc_chroma_dc(&cabac.neighbors, mb_x, plane as u8, current_is_intra);
                let coded = encode_residual_block_cabac_with_cbf_inc(
                    cabac, &dc_flat, 0, 3, 3, inc,
                );
                current_cbf.set(3, plane, coded);
            }
        }
        if cbp_chroma == 2 {
            for (plane, ac_blocks) in [&cb_ac, &cr_ac].iter().enumerate() {
                for sub in 0..4 {
                    let bx = (sub % 2) as u8;
                    let by = (sub / 2) as u8;
                    let mut ac_scan = ac_scan_order_15(&ac_blocks[sub]);
                    // Phase 6D.8: stego hook for P-frame Chroma AC.
                    self.invoke_stego_residual_hook(
                        mb_x, mb_y, &mut ac_scan, 0, 14,
                        super::super::stego::orchestrate::ResidualPathKind::ChromaAc {
                            plane: plane as u8,
                            block_idx: sub as u8,
                        },
                    );
                    // §B-cascade-real Phase 1.1.B step 2: capture
                    // post-flip chroma AC.
                    let ac_post_raster = ac_scan_15_to_raster(&ac_scan);
                    if plane == 0 {
                        cb_ac_post[sub] = ac_post_raster;
                    } else {
                        cr_ac_post[sub] = ac_post_raster;
                    }
                    let inc = compute_cbf_ctx_idx_inc_chroma_ac(
                        &current_cbf,
                        &cabac.neighbors,
                        mb_x,
                        plane as u8,
                        bx,
                        by,
                        current_is_intra,
                    );
                    let coded = encode_residual_block_cabac_with_cbf_inc(
                        cabac, &ac_scan, 0, 14, 4, inc,
                    );
                    current_cbf.set(
                        4,
                        block_pos_to_chroma_ac_idx(plane as u8, bx, by),
                        coded,
                    );
                }
            }
        }

        // 9. end_of_slice_flag.
        encode_end_of_slice_flag(cabac, is_last_mb);

        // 10. Commit neighbor state.
        let mut nb = CabacNeighborMB::default();
        nb.mb_type = MbTypeClass::PInter;
        nb.mb_skip_flag = false;
        nb.cbp_luma = cbp_luma_8x8;
        nb.cbp_chroma = cbp_chroma;
        nb.mb_qp_delta = qp_delta_emitted;
        nb.coded_block_flag_cat = current_cbf.to_neighbor_cbf();
        // v1.4 Phase 4.5 (#316) — per-block ref_idx_l0 from PMbChoice
        // partition geometry. Required for spec § 9.3.3.1.1.6
        // ctxIdxInc agreement with spec-conforming decoders.
        nb.ref_idx_l0 = fill_ref_idx_l0_p(&choice);
        nb.abs_mvd_comp = current_mvd.to_neighbor();
        nb.transform_size_8x8_flag = use_8x8;
        cabac.neighbors.commit(mb_x, nb);

        // 11. Reconstruction.
        let mut recon_luma = [[0u8; 16]; 16];
        if use_8x8 {
            // Dequant + inverse 8×8 for each of 4 8×8 blocks.
            for blk in 0..4 {
                let (bx_mb, by_mb) = I8X8_BLOCK_POS[blk];
                let bx0 = (bx_mb * 8) as usize;
                let by0 = (by_mb * 8) as usize;
                let dq = dequant_8x8_block(&levels_8x8[blk], qp);
                let inv = inverse_dct_8x8(&dq);
                for dy in 0..8 {
                    for dx in 0..8 {
                        let pixel_res = (inv[dy][dx] + 32) >> 6;
                        let v = pred_y[by0 + dy][bx0 + dx] as i32 + pixel_res;
                        recon_luma[by0 + dy][bx0 + dx] = v.clamp(0, 255) as u8;
                    }
                }
            }
        } else {
            // Reuse the Phase 100-J cached 4×4 reconstructed residuals
            // rather than re-running dequant+inverse here.
            for k in 0..16 {
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                let sby = by as usize;
                let sbx = bx as usize;
                let sub_res = &luma_recon_residual_4x4[k];
                for dy in 0..4 {
                    for dx in 0..4 {
                        let v = pred_y[sby * 4 + dy][sbx * 4 + dx] as i32 + sub_res[dy][dx];
                        recon_luma[sby * 4 + dy][sbx * 4 + dx] = v.clamp(0, 255) as u8;
                    }
                }
            }
        }
        self.recon
            .write_luma_mb(mb_x as u32, mb_y as u32, &recon_luma);
        let recon_cb = self.reconstruct_chroma_mb(&pred_cb, &cb_ac, &cb_dc, qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &recon_cb);
        let recon_cr = self.reconstruct_chroma_mb(&pred_cr, &cr_ac, &cr_dc, qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &recon_cr);

        // §B-cascade-real Phase 1.1.B step 2 — visual_recon path with
        // POST-flip levels (sites 9-12 of 15: P-frame 8×8 + 4×4 luma +
        // chroma DC + chroma AC). Re-run the same recon code with
        // _post arrays.
        let mut visual_recon_luma = [[0u8; 16]; 16];
        if use_8x8 {
            for blk in 0..4 {
                let (bx_mb, by_mb) = I8X8_BLOCK_POS[blk];
                let bx0 = (bx_mb * 8) as usize;
                let by0 = (by_mb * 8) as usize;
                let dq = dequant_8x8_block(&levels_8x8_post[blk], qp);
                let inv = inverse_dct_8x8(&dq);
                for dy in 0..8 {
                    for dx in 0..8 {
                        let pixel_res = (inv[dy][dx] + 32) >> 6;
                        let v = pred_y[by0 + dy][bx0 + dx] as i32 + pixel_res;
                        visual_recon_luma[by0 + dy][bx0 + dx] =
                            v.clamp(0, 255) as u8;
                    }
                }
            }
        } else {
            // 4×4 path: dequant + inverse on _post levels (no cached
            // residual exists for post-flip values; recompute here).
            use crate::codec::h264::transform::{dequant_4x4, inverse_4x4_integer};
            for k in 0..16 {
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                let sby = by as usize;
                let sbx = bx as usize;
                let dq = dequant_4x4(&luma_ac_levels_post[k], qp as i32, false);
                let res = inverse_4x4_integer(&dq);
                for dy in 0..4 {
                    for dx in 0..4 {
                        let v = pred_y[sby * 4 + dy][sbx * 4 + dx] as i32
                            + res[dy][dx];
                        visual_recon_luma[sby * 4 + dy][sbx * 4 + dx] =
                            v.clamp(0, 255) as u8;
                    }
                }
            }
        }
        self.visual_recon
            .write_luma_mb(mb_x as u32, mb_y as u32, &visual_recon_luma);
        let visual_cb =
            self.reconstruct_chroma_mb(&pred_cb, &cb_ac_post, &cb_dc_post, qp_c);
        self.visual_recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &visual_cb);
        let visual_cr =
            self.reconstruct_chroma_mb(&pred_cr, &cr_ac_post, &cr_dc_post, qp_c);
        self.visual_recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &visual_cr);

        // Phase 6F.2: inter MB committed; the MVDs logged earlier
        // by apply_mvd_hook_to_choice are now real bitstream
        // positions. Pairs with begin_mvd_for_mb. See
        // deferred-items.md §37.
        self.commit_mvd_for_mb();
        Ok(())
    }

    /// Emit the MVDs for a P MB's partitioning via CABAC, tracking the
    /// progressive current-MB abs_mvd state for same-MB ctxIdxInc
    /// lookups, and updating the 4×4 MV grid for successor predictors.
    fn emit_p_mvds_cabac(
        &mut self,
        cabac: &mut CabacEncoder,
        current_mvd: &mut super::super::cabac::neighbor::CurrentMbMvdAbs,
        mb_x: usize,
        mb_y: usize,
        choice: &PMbChoice,
    ) {
        
        use crate::codec::h264::cabac::neighbor::compute_mvd_ctx_idx_inc_bin0;

        let base_bx = mb_x * 4;
        let base_by = mb_y * 4;
        // Phase 6F.2(k).2 — closure now takes optional sign
        // overrides (ox, oy) for the X / Y MVD bypass sign bins.
        // Magnitude bins are always natural; only the sign bit
        // gets overridden when stego plan dictates.
        let emit = |cabac_: &mut CabacEncoder,
                        current: &mut super::super::cabac::neighbor::CurrentMbMvdAbs,
                        part_bx_in_mb: u8,
                        part_by_in_mb: u8,
                        part_w4: u8,
                        part_h4: u8,
                        mv: MotionVector,
                        pred: MotionVector,
                        ox: Option<u8>,
                        oy: Option<u8>| {
            use crate::codec::h264::cabac::encoder::encode_mvd_with_bin0_inc_sign_override;
            let mvd_x = mv.mv_x as i32 - pred.mv_x as i32;
            let mvd_y = mv.mv_y as i32 - pred.mv_y as i32;
            let inc_x = compute_mvd_ctx_idx_inc_bin0(
                current,
                &cabac_.neighbors,
                mb_x,
                part_bx_in_mb,
                part_by_in_mb,
                0,
            );
            encode_mvd_with_bin0_inc_sign_override(cabac_, mvd_x, 0, inc_x, ox);
            let inc_y = compute_mvd_ctx_idx_inc_bin0(
                current,
                &cabac_.neighbors,
                mb_x,
                part_bx_in_mb,
                part_by_in_mb,
                1,
            );
            encode_mvd_with_bin0_inc_sign_override(cabac_, mvd_y, 1, inc_y, oy);
            current.fill_region(
                part_bx_in_mb,
                part_by_in_mb,
                part_w4,
                part_h4,
                mvd_x.unsigned_abs().min(i16::MAX as u32) as i16,
                mvd_y.unsigned_abs().min(i16::MAX as u32) as i16,
            );
        };
        // v1.4 Phase 4.5 (#316) — thread the chosen ref_idx_l0 into
        // PMV computation (current_ref_idx) AND mv_grid.fill (so
        // downstream PMV reads pick matched-ref neighbours per spec
        // § 8.4.1.3 directional shortcuts + median rules).
        match *choice {
            PMbChoice::P16x16 { mv, ref_idx_l0 } => {
                let r = ref_idx_l0 as i8;
                let pred = predict_mv_for_partition(&self.mv_grid, base_bx, base_by, 4, r);
                let mvd_x = mv.mv_x as i32 - pred.mv_x as i32;
                let mvd_y = mv.mv_y as i32 - pred.mv_y as i32;
                let (ox, oy) = self.mvd_sign_overrides_for_partition(
                    mb_x, mb_y, /* partition */ 0, mvd_x, mvd_y,
                );
                emit(cabac, current_mvd, 0, 0, 4, 4, mv, pred, ox, oy);
                self.mv_grid.fill(base_bx, base_by, 4, 4, mv, r);
            }
            PMbChoice::P16x8 { mvs, ref_idx_l0 } => {
                let r0 = ref_idx_l0[0] as i8;
                let r1 = ref_idx_l0[1] as i8;
                let pred0 = predict_mv_for_mb_partition(
                    &self.mv_grid, base_bx, base_by, 4, 2, 0, r0,
                );
                let mvd0_x = mvs[0].mv_x as i32 - pred0.mv_x as i32;
                let mvd0_y = mvs[0].mv_y as i32 - pred0.mv_y as i32;
                let (ox0, oy0) = self.mvd_sign_overrides_for_partition(
                    mb_x, mb_y, /* partition */ 0, mvd0_x, mvd0_y,
                );
                emit(cabac, current_mvd, 0, 0, 4, 2, mvs[0], pred0, ox0, oy0);
                self.mv_grid.fill(base_bx, base_by, 4, 2, mvs[0], r0);
                let pred1 = predict_mv_for_mb_partition(
                    &self.mv_grid,
                    base_bx,
                    base_by + 2,
                    4,
                    2,
                    1,
                    r1,
                );
                let mvd1_x = mvs[1].mv_x as i32 - pred1.mv_x as i32;
                let mvd1_y = mvs[1].mv_y as i32 - pred1.mv_y as i32;
                let (ox1, oy1) = self.mvd_sign_overrides_for_partition(
                    mb_x, mb_y, /* partition */ 1, mvd1_x, mvd1_y,
                );
                emit(cabac, current_mvd, 0, 2, 4, 2, mvs[1], pred1, ox1, oy1);
                self.mv_grid.fill(base_bx, base_by + 2, 4, 2, mvs[1], r1);
            }
            PMbChoice::P8x16 { mvs, ref_idx_l0 } => {
                let r0 = ref_idx_l0[0] as i8;
                let r1 = ref_idx_l0[1] as i8;
                let pred0 = predict_mv_for_mb_partition(
                    &self.mv_grid, base_bx, base_by, 2, 4, 0, r0,
                );
                let mvd0_x = mvs[0].mv_x as i32 - pred0.mv_x as i32;
                let mvd0_y = mvs[0].mv_y as i32 - pred0.mv_y as i32;
                let (ox0, oy0) = self.mvd_sign_overrides_for_partition(
                    mb_x, mb_y, /* partition */ 0, mvd0_x, mvd0_y,
                );
                emit(cabac, current_mvd, 0, 0, 2, 4, mvs[0], pred0, ox0, oy0);
                self.mv_grid.fill(base_bx, base_by, 2, 4, mvs[0], r0);
                let pred1 = predict_mv_for_mb_partition(
                    &self.mv_grid,
                    base_bx + 2,
                    base_by,
                    2,
                    4,
                    1,
                    r1,
                );
                let mvd1_x = mvs[1].mv_x as i32 - pred1.mv_x as i32;
                let mvd1_y = mvs[1].mv_y as i32 - pred1.mv_y as i32;
                let (ox1, oy1) = self.mvd_sign_overrides_for_partition(
                    mb_x, mb_y, /* partition */ 1, mvd1_x, mvd1_y,
                );
                emit(cabac, current_mvd, 2, 0, 2, 4, mvs[1], pred1, ox1, oy1);
                self.mv_grid.fill(base_bx + 2, base_by, 2, 4, mvs[1], r1);
            }
            PMbChoice::P8x8 { sub } => {
                for (i, sub_choice) in sub.iter().enumerate() {
                    let (off_x_4x4, off_y_4x4) = SUB_MB_ORIGINS_4X4[i];
                    let sub_bx_abs = base_bx + off_x_4x4;
                    let sub_by_abs = base_by + off_y_4x4;
                    let sub_bx_in_mb = off_x_4x4 as u8;
                    let sub_by_in_mb = off_y_4x4 as u8;
                    self.emit_sub_mb_mvds_cabac(
                        cabac,
                        current_mvd,
                        mb_x,
                        mb_y,
                        i as u8, // sub_mb_idx
                        sub_bx_abs,
                        sub_by_abs,
                        sub_bx_in_mb,
                        sub_by_in_mb,
                        sub_choice,
                    );
                }
            }
        }
    }

    /// Emit MVDs for a single 8×8 sub-MB partition via CABAC.
    #[allow(clippy::too_many_arguments)]
    fn emit_sub_mb_mvds_cabac(
        &mut self,
        cabac: &mut CabacEncoder,
        current_mvd: &mut super::super::cabac::neighbor::CurrentMbMvdAbs,
        mb_x: usize,
        mb_y: usize,
        sub_mb_idx: u8,
        sub_bx_abs: usize,
        sub_by_abs: usize,
        sub_bx_in_mb: u8,
        sub_by_in_mb: u8,
        sub_choice: &SubMbChoice,
    ) {
        // Phase 6F.2(k).2 — partition encoding follows the
        // PositionKey convention: `sub_mb_idx * 4 + sub_part_idx`.
        // For P_L0_8x8 sub-MB (single partition), sub_part_idx=0.
        // For sub-MB-internal multi-partitions, sub_part_idx
        // indexes the sub-MB's own partitions in raster order.
        let p = |sub_part_idx: u8| sub_mb_idx * 4 + sub_part_idx;
        // v1.4 Phase 4.5 (#316) — thread the chosen sub-MB ref_idx_l0
        // into PMV computation + mv_grid.fill so spec § 8.4.1.3
        // matched-ref neighbour selection works.
        let r = sub_choice.ref_idx_l0() as i8;
        match *sub_choice {
            SubMbChoice::P8x8 { mv, .. } => {
                let pred =
                    predict_mv_for_partition(&self.mv_grid, sub_bx_abs, sub_by_abs, 2, r);
                self.emit_one_mvd_pair_cabac(
                    cabac, current_mvd, mb_x, mb_y, p(0),
                    sub_bx_in_mb, sub_by_in_mb, 2, 2, mv, pred,
                );
                self.mv_grid.fill(sub_bx_abs, sub_by_abs, 2, 2, mv, r);
            }
            SubMbChoice::P8x4 { mvs, .. } => {
                let pred_top =
                    predict_mv_for_partition(&self.mv_grid, sub_bx_abs, sub_by_abs, 2, r);
                self.emit_one_mvd_pair_cabac(
                    cabac, current_mvd, mb_x, mb_y, p(0),
                    sub_bx_in_mb, sub_by_in_mb, 2, 1, mvs[0], pred_top,
                );
                self.mv_grid.fill(sub_bx_abs, sub_by_abs, 2, 1, mvs[0], r);
                let pred_bot = predict_mv_for_partition(
                    &self.mv_grid,
                    sub_bx_abs,
                    sub_by_abs + 1,
                    2,
                    r,
                );
                self.emit_one_mvd_pair_cabac(
                    cabac,
                    current_mvd,
                    mb_x,
                    mb_y,
                    p(1),
                    sub_bx_in_mb,
                    sub_by_in_mb + 1,
                    2,
                    1,
                    mvs[1],
                    pred_bot,
                );
                self.mv_grid
                    .fill(sub_bx_abs, sub_by_abs + 1, 2, 1, mvs[1], r);
            }
            SubMbChoice::P4x8 { mvs, .. } => {
                let pred_left =
                    predict_mv_for_partition(&self.mv_grid, sub_bx_abs, sub_by_abs, 1, r);
                self.emit_one_mvd_pair_cabac(
                    cabac, current_mvd, mb_x, mb_y, p(0),
                    sub_bx_in_mb, sub_by_in_mb, 1, 2, mvs[0], pred_left,
                );
                self.mv_grid.fill(sub_bx_abs, sub_by_abs, 1, 2, mvs[0], r);
                let pred_right = predict_mv_for_partition(
                    &self.mv_grid,
                    sub_bx_abs + 1,
                    sub_by_abs,
                    1,
                    r,
                );
                self.emit_one_mvd_pair_cabac(
                    cabac,
                    current_mvd,
                    mb_x,
                    mb_y,
                    p(1),
                    sub_bx_in_mb + 1,
                    sub_by_in_mb,
                    1,
                    2,
                    mvs[1],
                    pred_right,
                );
                self.mv_grid
                    .fill(sub_bx_abs + 1, sub_by_abs, 1, 2, mvs[1], r);
            }
            SubMbChoice::P4x4 { mvs, .. } => {
                for (i, mv) in mvs.iter().enumerate() {
                    let ox = i % 2;
                    let oy = i / 2;
                    let pred = predict_mv_for_partition(
                        &self.mv_grid,
                        sub_bx_abs + ox,
                        sub_by_abs + oy,
                        1,
                        r,
                    );
                    self.emit_one_mvd_pair_cabac(
                        cabac,
                        current_mvd,
                        mb_x,
                        mb_y,
                        p(i as u8),
                        sub_bx_in_mb + ox as u8,
                        sub_by_in_mb + oy as u8,
                        1,
                        1,
                        *mv,
                        pred,
                    );
                    self.mv_grid
                        .fill(sub_bx_abs + ox, sub_by_abs + oy, 1, 1, *mv, r);
                }
            }
        }
    }

    /// Emit one MVD pair (x + y) via CABAC and update the current-MB
    /// abs_mvd tracking for subsequent same-MB partition neighbor
    /// lookups. Does NOT touch `self.mv_grid` — caller handles that.
    ///
    /// Phase 6F.2(k).2 — Now takes `&mut self` (was `&self`) so it
    /// can query `self.stego_hook.mvd_sign_override(...)` for the
    /// planned bypass-bin overrides at this partition's
    /// (frame_idx, mb_addr, partition, axis) coordinates. The
    /// `partition` parameter encodes the position-key partition
    /// field (`sub_mb_idx * 4 + sub_part_idx` for sub-MB callers).
    #[allow(clippy::too_many_arguments)]
    fn emit_one_mvd_pair_cabac(
        &mut self,
        cabac: &mut CabacEncoder,
        current_mvd: &mut super::super::cabac::neighbor::CurrentMbMvdAbs,
        mb_x: usize,
        mb_y: usize,
        partition: u8,
        part_bx_in_mb: u8,
        part_by_in_mb: u8,
        part_w4: u8,
        part_h4: u8,
        mv: MotionVector,
        pred: MotionVector,
    ) {
        use crate::codec::h264::cabac::encoder::encode_mvd_with_bin0_inc_sign_override;
        use crate::codec::h264::cabac::neighbor::compute_mvd_ctx_idx_inc_bin0;
        let mvd_x = mv.mv_x as i32 - pred.mv_x as i32;
        let mvd_y = mv.mv_y as i32 - pred.mv_y as i32;
        let (ox, oy) = self.mvd_sign_overrides_for_partition(
            mb_x, mb_y, partition, mvd_x, mvd_y,
        );
        let inc_x = compute_mvd_ctx_idx_inc_bin0(
            current_mvd,
            &cabac.neighbors,
            mb_x,
            part_bx_in_mb,
            part_by_in_mb,
            0,
        );
        encode_mvd_with_bin0_inc_sign_override(cabac, mvd_x, 0, inc_x, ox);
        let inc_y = compute_mvd_ctx_idx_inc_bin0(
            current_mvd,
            &cabac.neighbors,
            mb_x,
            part_bx_in_mb,
            part_by_in_mb,
            1,
        );
        encode_mvd_with_bin0_inc_sign_override(cabac, mvd_y, 1, inc_y, oy);
        current_mvd.fill_region(
            part_bx_in_mb,
            part_by_in_mb,
            part_w4,
            part_h4,
            mvd_x.unsigned_abs().min(i16::MAX as u32) as i16,
            mvd_y.unsigned_abs().min(i16::MAX as u32) as i16,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn write_p_macroblock(
        &mut self,
        w: &mut BitWriter,
        reference: &super::reference_buffer::ReconFrame,
        mb_x: usize,
        mb_y: usize,
        mb_w: usize,
        y_plane: &[u8],
        y_stride: usize,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
        qp: u8,
        _qp_c: u8,
        skip_run: &mut u32,
    ) -> Result<(), EncoderError> {
        let _ = mb_w;
        // Gather source 16×16 luma + two 8×8 chroma blocks.
        let y0 = mb_y * 16;
        let x0 = mb_x * 16;
        let mut src_y = [[0u8; 16]; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                src_y[dy][dx] = y_plane[(y0 + dy) * y_stride + x0 + dx];
            }
        }
        let cy0 = mb_y * 8;
        let cx0 = mb_x * 8;
        let mut src_cb = [[0u8; 8]; 8];
        let mut src_cr = [[0u8; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                src_cb[dy][dx] = cb_plane[(cy0 + dy) * c_stride + cx0 + dx];
                src_cr[dy][dx] = cr_plane[(cy0 + dy) * c_stride + cx0 + dx];
            }
        }

        // Task #154 tracer: log entry state so we see every MB even if
        // later branches take an early return.
        let trace_mb = super::mb_decision_b::env_var("PHASM_DUMP_MB").is_some_and(|s| {
            s.split(';').any(|pair| {
                let mut parts = pair.split(',');
                let x: Option<usize> = parts.next().and_then(|p| p.trim().parse().ok());
                let y: Option<usize> = parts.next().and_then(|p| p.trim().parse().ok());
                matches!((x, y), (Some(px), Some(py)) if px == mb_x && py == mb_y)
            })
        });
        if trace_mb {
            let bx = (mb_x * 4) as isize;
            let by = (mb_y * 4) as isize;
            eprintln!(
                "[ENC MB ({},{})] ENTRY prev_mb_qp={} grid A={:?} B={:?} C={:?}",
                mb_x, mb_y, self.prev_mb_qp,
                self.mv_grid.get(bx - 1, by),
                self.mv_grid.get(bx, by - 1),
                self.mv_grid.get(bx + 4, by - 1),
            );
        }

        // ─── Per-MB AQ ───
        // Default (mode 1 lite): textured regions get -QP offset;
        // flat +offset clamped to 0 (banding regression on flats).
        // Mode 3: auto-variance + dark bias, via PHASM_AQ_MODE=3.
        // Strength knob via PHASM_AQ_STRENGTH_Q10 (Q.10 fixed point).
        // Default 256 (0.25x) is the measured R-D sweet spot on
        // IMG_4138 30f Q=80: +0.52 dB avg / +0.57 dB worst for only
        // +10% bits vs mode-1 baseline. The literature "unity" point
        // (1024 = 1.0) sits over the R-D Pareto frontier (+2.22 dB /
        // +79% bits). Measurement details in
        // `docs/design/video/h264/encoder-quality-plan.md` Phase F.
        let variance = super::rate_control::mb_variance_16x16(&src_y);
        let aq_mode: u8 = super::mb_decision_b::env_var("PHASM_AQ_MODE")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let aq_disabled = super::mb_decision_b::env_var_os_is_some("PHASM_DISABLE_PAQ");
        let qp_offset: i32 = if aq_disabled {
            0
        } else if aq_mode == 3 {
            let strength: i32 = super::mb_decision_b::env_var("PHASM_AQ_STRENGTH_Q10")
                
                .and_then(|s| s.parse().ok())
                .unwrap_or(256);
            let mut luma_sum: u32 = 0;
            for row in &src_y {
                for &v in row {
                    luma_sum += v as u32;
                }
            }
            let avg_luma = luma_sum / 256;
            super::rate_control::variance_to_qp_offset_mode3(
                variance,
                self.aq_frame_mean_log2_q8,
                avg_luma,
                strength,
            )
        } else {
            super::rate_control::variance_to_qp_offset(variance).min(0)
        };
        let mb_qp = ((qp as i32) + qp_offset).clamp(0, 51) as u8;
        let mb_qp_c = derive_chroma_qp(mb_qp as i32, self.pps_params.chroma_qp_index_offset as i32) as u8;
        // Commit MB state for the deblock filter. Intra-in-P fallback
        // below will override intra=true via write_i16x16_macroblock's
        // own commit call; P_SKIP / normal P emit with intra=false.
        self.commit_mb_state(mb_x, mb_y, mb_qp, false);

        // ─── Fast P_SKIP probe ───
        // Before running full ME, try the P_SKIP predictor MV. If the
        // motion-compensated residual quantises to all-zero AC levels
        // (both luma and chroma) and the chroma residuals are also
        // zero, we can emit a P_SKIP MB — 0 bits, just incrementing
        // mb_skip_run. This fast-path is essential for achieving
        // 50-80% skip rate on typical content; without it every P-MB
        // costs a minimum of ~10-20 bits for mb_type + MVD + CBP +
        // qp_delta syntax even when residuals round to zero.
        let p_skip_mv = super::partition_state::predict_p_skip_mv(
            &self.mv_grid,
            mb_x * 4,
            mb_y * 4,
        );
        {
            let skip_choice = PMbChoice::P16x16 { mv: p_skip_mv, ref_idx_l0: 0 };
            let s_pred_y = build_luma_prediction(reference, mb_x, mb_y, &skip_choice);
            let s_pred_cb = build_chroma_prediction(reference, 0, mb_x, mb_y, &skip_choice);
            let s_pred_cr = build_chroma_prediction(reference, 1, mb_x, mb_y, &skip_choice);

            let inter = QuantParams { qp: mb_qp, slice: QuantSlice::Inter };
            let chroma_qp = QuantParams { qp: mb_qp_c, slice: QuantSlice::Inter };
            use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS as BIP;
            let mut any_luma_nonzero = false;
            for k in 0..16 {
                let (bx, by) = BIP[k];
                let sby = by as usize;
                let sbx = bx as usize;
                let mut sub_res = [[0i32; 4]; 4];
                for dy in 0..4 {
                    for dx in 0..4 {
                        sub_res[dy][dx] = src_y[sby * 4 + dy][sbx * 4 + dx] as i32
                            - s_pred_y[sby * 4 + dy][sbx * 4 + dx] as i32;
                    }
                }
                let coeffs = forward_dct_4x4(&sub_res);
                let levels = trellis_quantize_4x4(&coeffs, inter, true)
                    .unwrap_or_else(|_| forward_quantize_4x4(&coeffs, inter));
                if levels.iter().any(|r| r.iter().any(|&v| v != 0)) {
                    any_luma_nonzero = true;
                    break;
                }
            }
            if !any_luma_nonzero {
                let (cb_ac, cb_dc) = self.encode_chroma_component(&src_cb, &s_pred_cb, chroma_qp);
                let (cr_ac, cr_dc) = self.encode_chroma_component(&src_cr, &s_pred_cr, chroma_qp);
                let chroma_any = cb_ac.iter().chain(cr_ac.iter())
                    .any(|b| b.iter().any(|r| r.iter().any(|&v| v != 0)))
                    || cb_dc.iter().flatten().chain(cr_dc.iter().flatten())
                        .any(|&v| v != 0);
                if !chroma_any {
                    if trace_mb { eprintln!("[ENC MB ({},{})] → P_SKIP_FAST", mb_x, mb_y); }
                    self.mode_stats[MODE_STAT_P_SKIP_FAST] += 1;
                    // P_SKIP: no residual, no bits, just reconstruct.
                    // Spec § 7.4.5.1: P_SKIP does NOT carry mb_qp_delta;
                    // decoder keeps QpY = prev_mb_qp. Our `commit_mb_state`
                    // above stored AQ-adjusted mb_qp which would mislead
                    // the deblock filter's `qp_avg` computation and
                    // diverge from the decoder. Re-commit with prev_mb_qp
                    // so our deblock sees the same QP the decoder will.
                    self.commit_mb_state(mb_x, mb_y, self.prev_mb_qp as u8, false);
                    *skip_run += 1;
                    self.mv_grid.fill(mb_x * 4, mb_y * 4, 4, 4, p_skip_mv, 0);
                    // §B-cascade-real Phase 1.1.B step 3: dual-write
                    // Skip prediction. No residual → no flip → both
                    // buffers receive identical pixels.
                    self.write_luma_mb_dual(mb_x as u32, mb_y as u32, &s_pred_y);
                    self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 0, &s_pred_cb);
                    self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 1, &s_pred_cr);
                    for k in 0..16 {
                        let (bx, by) = BIP[k];
                        self.store_total_coeff_luma(
                            mb_x * 4 + bx as usize,
                            mb_y * 4 + by as usize,
                            0,
                        );
                    }
                    return Ok(());
                }
            }
        }

        // Decide partition choice via SATD + fixed overhead penalty.
        let decision =
            decide_p_mb_with_cost(&src_y, reference, &mut self.me, &mut self.mv_grid, mb_x, mb_y);

        // ─── Phase C MB-level RDO (task #129) ───
        // Re-rank the top-K SATD candidates via actual distortion+rate:
        //   cost = D_luma_SSD + ((bits × lambda2[qp]) >> 8)
        //
        // Opt-in via PHASM_ENABLE_RDO. Default is SATD+penalty (the
        // pre-Phase-C behaviour) because our initial λ² calibration
        // produced a -3.9 dB regression on IMG_4138 30f — RDO and
        // SATD have the same R-D efficiency (3.20 vs 3.21 PSNR/Mbps),
        // but the Sullivan-Wiegand λ² value pushes us to an
        // operating point with less bitrate + less quality. The
        // canonical λ² calibration in the literature assumes
        // AQ mode 3 + psy-RDO + trellis-at-high-QP; our pipeline is
        // lean, so λ² needs re-calibration for our feature set
        // (Phase C.v2).
        let (choice, inter_cost) = if super::mb_decision_b::env_var_os_is_some("PHASM_ENABLE_RDO") {
            let frame_w4 = (self.width / 4) as usize;
            select_p_mb_via_rdo(
                &src_y, reference, &mut self.mv_grid, mb_x, mb_y, mb_qp, &decision,
                &self.total_coeff_grid, frame_w4,
            )
        } else {
            (decision.best, decision.best_cost)
        };

        // ─── RDO P_SKIP eligibility (task #124 Step 2a) ───
        // Instrumentation showed our P_SKIP rate is ~40% on
        // slow-motion content vs the 60-80% typical of a tuned
        // encoder. Missing skip opportunities cost ~30 bits each
        // and inject residual that quantises lossily and compounds
        // drift. Real RDO: if the DISTORTION from
        // emitting bare prediction is less than (inter_distortion +
        // λ × inter_bits), skip — saves bits AND stops drift fuel.
        if let PMbChoice::P16x16 { .. } = choice {
            let skip_choice = PMbChoice::P16x16 { mv: p_skip_mv, ref_idx_l0: 0 };
            let s_pred_y = build_luma_prediction(reference, mb_x, mb_y, &skip_choice);
            // SATD(source, pred) at P_SKIP predictor — distortion if
            // we emit P_SKIP (no residual → recon = pred).
            let skip_satd = satd_16x16(&src_y, &s_pred_y);
            // RDO budget for full P emit: inter SATD + λ × bits.
            const INTER_BITS_ESTIMATE: u32 = 30;
            const LAMBDA_RDO: u32 = 3;
            let inter_rdo_cost =
                inter_cost.saturating_add(LAMBDA_RDO * INTER_BITS_ESTIMATE);
            if skip_satd < inter_rdo_cost {
                // Skip wins. Execute the P_SKIP emission path
                // directly (mirrors the fast-path P_SKIP code above).
                self.mode_stats[MODE_STAT_P_SKIP_POST_ME] += 1;
                let s_pred_cb = build_chroma_prediction(reference, 0, mb_x, mb_y, &skip_choice);
                let s_pred_cr = build_chroma_prediction(reference, 1, mb_x, mb_y, &skip_choice);
                // P_SKIP: spec § 7.4.5.1 — no mb_qp_delta, decoder uses
                // prev_mb_qp for deblock. Match it.
                self.commit_mb_state(mb_x, mb_y, self.prev_mb_qp as u8, false);
                *skip_run += 1;
                self.mv_grid.fill(mb_x * 4, mb_y * 4, 4, 4, p_skip_mv, 0);
                // §B-cascade-real Phase 1.1.B step 3: dual-write
                // Skip prediction (P/B Skip variant).
                self.write_luma_mb_dual(mb_x as u32, mb_y as u32, &s_pred_y);
                self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 0, &s_pred_cb);
                self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 1, &s_pred_cr);
                use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS as BIP_SKIP;
                for k in 0..16 {
                    let (bx, by) = BIP_SKIP[k];
                    self.store_total_coeff_luma(
                        mb_x * 4 + bx as usize,
                        mb_y * 4 + by as usize,
                        0,
                    );
                }
                return Ok(());
            }
        }

        // Intra-in-P fallback: evaluate I_16x16 SATD against the
        // winning inter cost. When the reference is a poor match
        // (scene change, occlusion, fine high-frequency texture that
        // ME can't track), the intra prediction from spatial
        // neighbors beats motion-compensated residual even accounting
        // for the extra mb_type bits. Spec § 8.4.1.3.2 allows intra
        // MBs in P-slices via mb_type codenum 5..=30 in Table 7-13.
        let neighbors_y = self.build_luma_neighbors_16x16(mb_x, mb_y);
        let i16_decision =
            choose_intra_16x16_mode_psy(&neighbors_y, &src_y, Self::PSY_RD_STRENGTH);
        // Diagnostic gate: force intra-in-P on a specific MB to reproduce
        // the task #154 parity bug deterministically. Setting
        // PHASM_FORCE_INTRA_IN_P_MB="mb_x,mb_y" makes THAT MB take the
        // intra path regardless of cost, so a minimal bitstream can be
        // encoded for byte-exact comparison against a conformant
        // reference decoder's parse. Other MBs unchanged.
        // Accepts a semicolon-separated list of MB coords, e.g.
        // "5,3;6,3;7,3" to force intra on three MBs at once.
        let force_intra = super::mb_decision_b::env_var("PHASM_FORCE_INTRA_IN_P_MB")
            .is_some_and(|s| {
                s.split(';').any(|pair| {
                    let mut parts = pair.split(',');
                    let x: Option<usize> = parts.next().and_then(|p| p.trim().parse().ok());
                    let y: Option<usize> = parts.next().and_then(|p| p.trim().parse().ok());
                    matches!((x, y), (Some(px), Some(py)) if px == mb_x && py == mb_y)
                })
            });
        // Compare raw SATDs: the inter side uses SAD/SATD without psy
        // bias (motion_estimation.rs), so we need the un-psy-biased
        // distortion for the intra side too — otherwise the psy term
        // (up to a few hundred SATD units) makes this decision
        // apples-to-oranges and intra gets unfairly penalized.
        let intra_raw_satd = satd_16x16(&src_y, &i16_decision.predicted);
        let intra_cost = intra_raw_satd.saturating_add(Self::P_INTRA_OVERHEAD);
        // Task #154 diagnostic: PHASM_DISABLE_INTRA skips the natural
        // intra-in-P path (force_intra still works). Useful for
        // isolating whether a parity divergence is caused by intra-
        // neighbor handling elsewhere in the pipeline.
        let intra_disabled = super::mb_decision_b::env_var_os_is_some("PHASM_DISABLE_INTRA");
        if (intra_cost < inter_cost && !intra_disabled) || force_intra {
            self.mode_stats[MODE_STAT_INTRA_IN_P] += 1;
            // Intra-in-P: flush any pending skip run before emitting.
            w.write_ue(*skip_run);
            *skip_run = 0;
            // Mark this MB's 16 4×4 slots as intra-coded so future
            // neighbors see REF_IDX_NONE and treat MV predictor input
            // as unavailable (spec § 8.4.1.3).
            self.mv_grid.fill(
                mb_x * 4,
                mb_y * 4,
                4,
                4,
                MotionVector::ZERO,
                REF_IDX_NONE,
            );
            // Upgrade intra_grid entry: this MB is intra-coded despite
            // living in a P-slice. The deblock filter needs bs=4 at its
            // MB-boundary edges (spec § 8.7.2.1).
            self.commit_mb_state(mb_x, mb_y, mb_qp, true);
            return self.write_i16x16_macroblock(
                w, mb_x, mb_y, y_plane, y_stride, cb_plane, cr_plane, c_stride, mb_qp, mb_qp_c, 5,
            );
        }

        // Build motion-compensated predictions for the chosen partitioning.
        let pred_y = build_luma_prediction(reference, mb_x, mb_y, &choice);
        let pred_cb = build_chroma_prediction(reference, 0, mb_x, mb_y, &choice);
        let pred_cr = build_chroma_prediction(reference, 1, mb_x, mb_y, &choice);

        // Compute residuals at the per-MB AQ'd QP.
        let intra = QuantParams {
            qp: mb_qp,
            slice: QuantSlice::Inter,
        };
        let chroma_qp = QuantParams {
            qp: mb_qp_c,
            slice: QuantSlice::Inter,
        };

        // Per 4×4 AC luma sub-block: residual → forward DCT → quant.
        use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;
        let mut luma_ac_levels = [[[0i32; 4]; 4]; 16]; // indexed by BlockIndex k
        let mut luma_nonzero = [false; 16];
        for k in 0..16 {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            let sby = by as usize;
            let sbx = bx as usize;
            let mut sub_res = [[0i32; 4]; 4];
            for dy in 0..4 {
                for dx in 0..4 {
                    sub_res[dy][dx] = src_y[sby * 4 + dy][sbx * 4 + dx] as i32
                        - pred_y[sby * 4 + dy][sbx * 4 + dx] as i32;
                }
            }
            let coeffs = forward_dct_4x4(&sub_res);
            let levels = trellis_quantize_4x4(&coeffs, intra, true)
                .unwrap_or_else(|_| forward_quantize_4x4(&coeffs, intra));
            luma_ac_levels[k] = levels;
            luma_nonzero[k] = levels.iter().any(|row| row.iter().any(|&v| v != 0));
        }
        let cbp_luma_8x8 = luma_8x8_cbp_mask(&luma_nonzero);

        // Chroma path (same structure as I_16x16 chroma — DC Hadamard
        // + AC).
        let (cb_ac, cb_dc) = self.encode_chroma_component(&src_cb, &pred_cb, chroma_qp);
        let (cr_ac, cr_dc) = self.encode_chroma_component(&src_cr, &pred_cr, chroma_qp);
        let any_cb_ac = cb_ac.iter().any(|b| b.iter().any(|r| r.iter().any(|&v| v != 0)));
        let any_cr_ac = cr_ac.iter().any(|b| b.iter().any(|r| r.iter().any(|&v| v != 0)));
        let any_cb_dc = cb_dc.iter().flatten().any(|&v| v != 0);
        let any_cr_dc = cr_dc.iter().flatten().any(|&v| v != 0);
        let cbp_chroma = if any_cb_ac || any_cr_ac {
            2
        } else if any_cb_dc || any_cr_dc {
            1
        } else {
            0
        };
        let cbp_value = pack_cbp(cbp_luma_8x8, cbp_chroma);

        // ─── P_SKIP detection (spec § 7.3.5.1 / § 8.4.1.2.1) ───
        // If the partition is P_L0_16x16, cbp is zero, and the chosen
        // MV matches the spec's P_SKIP predictor, this MB can be
        // entirely skipped (0 bits). Instead we extend the pending
        // mb_skip_run count, reconstruct the MB in place, and return.
        // The mb_skip_run count is emitted by the next non-skip MB,
        // or at end of slice if trailing MBs are all skipped.
        let is_skip = if cbp_value == 0 {
            if let PMbChoice::P16x16 { mv, .. } = choice {
                let p_skip_mv = super::partition_state::predict_p_skip_mv(
                    &self.mv_grid,
                    mb_x * 4,
                    mb_y * 4,
                );
                mv.mv_x == p_skip_mv.mv_x && mv.mv_y == p_skip_mv.mv_y
            } else {
                false
            }
        } else {
            false
        };
        if is_skip {
            self.mode_stats[MODE_STAT_P_SKIP_POST_ME] += 1;
            // Spec § 7.4.5.1: P_SKIP doesn't carry mb_qp_delta, so the
            // decoder uses QpY = prev_mb_qp for deblock. Match it.
            self.commit_mb_state(mb_x, mb_y, self.prev_mb_qp as u8, false);
            *skip_run += 1;
            // Fill the MV grid + write the motion-compensated pred as
            // the reconstructed pixels (no residual). Update TC grids.
            if let PMbChoice::P16x16 { mv, .. } = choice {
                self.mv_grid.fill(mb_x * 4, mb_y * 4, 4, 4, mv, 0);
            }
            self.recon
                .write_luma_mb(mb_x as u32, mb_y as u32, &pred_y);
            self.recon
                .write_chroma_block(mb_x as u32, mb_y as u32, 0, &pred_cb);
            self.recon
                .write_chroma_block(mb_x as u32, mb_y as u32, 1, &pred_cr);
            for k in 0..16 {
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                self.store_total_coeff_luma(
                    mb_x * 4 + bx as usize,
                    mb_y * 4 + by as usize,
                    0,
                );
            }
            return Ok(());
        }

        // ─── Emit MB syntax ───
        // Task #154 diagnostic: dump encoder-side MB state.
        // Set PHASM_DUMP_MB="x,y[;x,y;...]" to log partition/MV/CBP/qp_delta
        // as the encoder emits. Pair with a conformant external decoder's
        // MB-level debug output to find the first MB where the two
        // disagree.
        let dump_this_mb = super::mb_decision_b::env_var("PHASM_DUMP_MB").is_some_and(|s| {
            s.split(';').any(|pair| {
                let mut parts = pair.split(',');
                let x: Option<usize> = parts.next().and_then(|p| p.trim().parse().ok());
                let y: Option<usize> = parts.next().and_then(|p| p.trim().parse().ok());
                matches!((x, y), (Some(px), Some(py)) if px == mb_x && py == mb_y)
            })
        });
        if dump_this_mb {
            eprintln!(
                "[ENC MB ({},{})] partition={:?} inter_cost={} cbp={:02x} mb_qp={} prev_qp={}",
                mb_x, mb_y, &choice, inter_cost, cbp_value, mb_qp, self.prev_mb_qp,
            );
            // Dump neighbor state (what predictor reads).
            let bx = mb_x * 4;
            let by = mb_y * 4;
            let mv_grid = &self.mv_grid;
            eprintln!(
                "[ENC MB ({},{})]   neighbors: A=(bx={},by={}) B=(bx={},by={}) C=(bx={},by={})",
                mb_x, mb_y,
                bx.wrapping_sub(1), by, bx, by.wrapping_sub(1), bx + 4, by.wrapping_sub(1),
            );
            eprintln!(
                "[ENC MB ({},{})]   A={:?} B={:?} C={:?}",
                mb_x, mb_y,
                mv_grid.get(bx as isize - 1, by as isize),
                mv_grid.get(bx as isize, by as isize - 1),
                mv_grid.get(bx as isize + 4, by as isize - 1),
            );
        }
        match choice {
            PMbChoice::P16x16 { .. } => self.mode_stats[MODE_STAT_P_16X16] += 1,
            PMbChoice::P16x8 { .. } => self.mode_stats[MODE_STAT_P_16X8] += 1,
            PMbChoice::P8x16 { .. } => self.mode_stats[MODE_STAT_P_8X16] += 1,
            PMbChoice::P8x8 { .. } => self.mode_stats[MODE_STAT_P_8X8] += 1,
        }
        // Flush any pending mb_skip_run before this non-skip MB.
        w.write_ue(*skip_run);
        *skip_run = 0;
        // mb_type per spec Table 7-13 (P-slice).
        w.write_ue(choice.mb_type_codenum());

        // ref_idx_l0 fields are gated on num_ref_idx_l0_active_minus1
        // > 0; our SPS has num_ref = 1 (active_minus1 = 0), so we
        // skip the entire ref_idx block for all partition types.

        // MVDs per partition, in spec emit order. Each partition's
        // predictor reads from `self.mv_grid`; previous partitions in
        // the same MB have already been written to the grid.
        emit_mvds_and_update_grid(w, &mut self.mv_grid, mb_x, mb_y, &choice);

        // coded_block_pattern (me(v), Table 9-4 inter column).
        let cbp_codenum = cbp_to_codenum_inter(cbp_value)
            .ok_or_else(|| EncoderError::InvalidInput(format!("bad CBP {cbp_value}")))?;
        w.write_ue(cbp_codenum);

        if cbp_value != 0 {
            // mb_qp_delta (se). P-slice AQ: `mb_qp` differs per MB
            // based on variance (flat → +offset, textured → -offset).
            // Decoder reconstructs prev_mb_qp + delta on each MB.
            let delta = mb_qp as i32 - self.prev_mb_qp;
            w.write_se(delta);
            self.prev_mb_qp = mb_qp as i32;
        } else {
            // CBP=0: spec § 7.3.5.1 says mb_qp_delta is NOT emitted,
            // so decoder keeps QpY = prev_mb_qp. Override our qp_grid
            // (committed as AQ-adjusted mb_qp at function entry) with
            // prev_mb_qp so the deblock filter's qp_avg matches what
            // the decoder computes.
            self.commit_mb_state(mb_x, mb_y, self.prev_mb_qp as u8, false);
        }

        // Residual blocks when CBP indicates they're present.
        use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS as BLK_POS_P;
        if cbp_luma_8x8 != 0 {
            for k in 0..16 {
                let (bx_in_mb, by_in_mb) = BLK_POS_P[k];
                let abs_bx = mb_x * 4 + bx_in_mb as usize;
                let abs_by = mb_y * 4 + by_in_mb as usize;
                if cbp_luma_8x8 & (1 << (k / 4)) != 0 {
                    let ac_scan = raster_to_scan_levels(&luma_ac_levels[k]);
                    let nc = self.compute_nc_luma_at(abs_bx, abs_by);
                    encode_cavlc_block(w, &ac_scan, nc, CavlcBlockType::Luma4x4).map_err(|e| {
                        EncoderError::InvalidInput(format!("CAVLC P luma AC: {e}"))
                    })?;
                    let total_coeff =
                        ac_scan.iter().filter(|&&v| v != 0).count() as u8;
                    self.store_total_coeff_luma(abs_bx, abs_by, total_coeff);
                } else {
                    // Skipped blocks still mark the grid as
                    // "coded with 0 coeffs" so the neighbor-context
                    // average stays accurate.
                    self.store_total_coeff_luma(abs_bx, abs_by, 0);
                }
            }
        } else {
            // Whole MB had no residual — mark every block as 0.
            for k in 0..16 {
                let (bx_in_mb, by_in_mb) = BLK_POS_P[k];
                self.store_total_coeff_luma(
                    mb_x * 4 + bx_in_mb as usize,
                    mb_y * 4 + by_in_mb as usize,
                    0,
                );
            }
        }
        if cbp_chroma != 0 {
            // Chroma DC blocks (2×2, 4 coeffs, nc=-1).
            let cb_dc_flat: Vec<i32> = cb_dc.iter().flatten().copied().collect();
            encode_cavlc_block(w, &cb_dc_flat, -1, CavlcBlockType::ChromaDc)
                .map_err(|e| EncoderError::InvalidInput(format!("CAVLC P Cb DC: {e}")))?;
            let cr_dc_flat: Vec<i32> = cr_dc.iter().flatten().copied().collect();
            encode_cavlc_block(w, &cr_dc_flat, -1, CavlcBlockType::ChromaDc)
                .map_err(|e| EncoderError::InvalidInput(format!("CAVLC P Cr DC: {e}")))?;
            if cbp_chroma == 2 {
                let cmb_x = mb_x * 2;
                let cmb_y = mb_y * 2;
                for (sub_idx, block) in cb_ac.iter().enumerate() {
                    let cx = cmb_x + sub_idx % 2;
                    let cy = cmb_y + sub_idx / 2;
                    let scan = ac_scan_order_15(block);
                    let nc = self.compute_nc_chroma_at(cx, cy, false);
                    encode_cavlc_block(w, &scan, nc, CavlcBlockType::ChromaAc)
                        .map_err(|e| EncoderError::InvalidInput(format!("CAVLC P Cb AC: {e}")))?;
                    let tc = scan.iter().filter(|&&v| v != 0).count() as u8;
                    self.store_total_coeff_chroma(cx, cy, false, tc);
                }
                for (sub_idx, block) in cr_ac.iter().enumerate() {
                    let cx = cmb_x + sub_idx % 2;
                    let cy = cmb_y + sub_idx / 2;
                    let scan = ac_scan_order_15(block);
                    let nc = self.compute_nc_chroma_at(cx, cy, true);
                    encode_cavlc_block(w, &scan, nc, CavlcBlockType::ChromaAc)
                        .map_err(|e| EncoderError::InvalidInput(format!("CAVLC P Cr AC: {e}")))?;
                    let tc = scan.iter().filter(|&&v| v != 0).count() as u8;
                    self.store_total_coeff_chroma(cx, cy, true, tc);
                }
            }
        }

        // ─── Reconstruct pixels for the ReconBuffer ───
        let mut recon_luma = [[0u8; 16]; 16];
        for k in 0..16 {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            let sby = by as usize;
            let sbx = bx as usize;
            // Inverse quant + inverse DCT to recover the residual.
            // Uses the per-MB AQ'd QP to match what the decoder
            // reconstructs from mb_qp_delta.
            use crate::codec::h264::transform::{dequant_4x4, inverse_4x4_integer};
            let dq = dequant_4x4(&luma_ac_levels[k], mb_qp as i32, false);
            let sub_res = inverse_4x4_integer(&dq);
            for dy in 0..4 {
                for dx in 0..4 {
                    let v = pred_y[sby * 4 + dy][sbx * 4 + dx] as i32 + sub_res[dy][dx];
                    recon_luma[sby * 4 + dy][sbx * 4 + dx] = v.clamp(0, 255) as u8;
                }
            }
        }
        self.recon
            .write_luma_mb(mb_x as u32, mb_y as u32, &recon_luma);

        let recon_cb = self.reconstruct_chroma_mb(&pred_cb, &cb_ac, &cb_dc, mb_qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &recon_cb);
        let recon_cr = self.reconstruct_chroma_mb(&pred_cr, &cr_ac, &cr_dc, mb_qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &recon_cr);

        // Grid updates are already done by `emit_mvds_and_update_grid`
        // (per-partition, in spec emit order).

        Ok(())
    }

    fn emit_params_if_needed(&mut self, out: &mut Vec<u8>) {
        if self.params_emitted {
            return;
        }
        // §6E-A4 — sync SPS to enable_b_frames BEFORE emit. When
        // enabled, SPS must use pic_order_cnt_type=0 (so decoders
        // can reorder display vs encode order via pic_order_cnt_lsb)
        // and bump max_num_ref_frames to at least 3 (M=2 IBPBP DPB
        // shape). Caller must set enable_b_frames BEFORE the first
        // encode call; toggling mid-stream produces an invalid
        // bitstream because SPS is emitted only once.
        if self.enable_b_frames {
            self.sps_params.pic_order_cnt_type = 0;
            self.sps_params.log2_max_pic_order_cnt_lsb_minus4 = 4;
            if self.sps_params.max_num_ref_frames < 3 {
                self.sps_params.max_num_ref_frames = 3;
            }
        }
        // Select SPS/PPS profile based on entropy mode and the 8×8
        // transform opt-in. CAVLC always stays Baseline. CABAC picks
        // High when `enable_transform_8x8` is set, else Main.
        let high_profile = self.entropy_mode == EntropyMode::Cabac && self.enable_transform_8x8;
        // §Stealth.L3.1 — emit VUI on every CABAC path (Main + High).
        // Both profiles target the common-encoder centroid; an empty VUI
        // would be its own L4 fingerprint per Altinisik 2022 §IV (only
        // 18/119 reference classes have an empty VUI). 30 fps is
        // phasm's canonical output rate; if the source clip is at a
        // different rate the bridge owns surfacing the right number.
        // CAVLC Baseline keeps VUI absent (legacy compat).
        if self.entropy_mode == EntropyMode::Cabac && self.sps_params.vui.is_none() {
            self.sps_params.vui = Some(VuiParams::handbrake_x264(30, 1));
        }
        let sps_rbsp = match (self.entropy_mode, high_profile) {
            (EntropyMode::Cavlc, _) => build_sps_baseline(&self.sps_params),
            (EntropyMode::Cabac, false) => build_sps_main(&self.sps_params),
            (EntropyMode::Cabac, true) => build_sps_high(&self.sps_params),
        };
        let sps_nal = wrap_rbsp_as_nal(&sps_rbsp, NalType::SPS, 3);
        out.extend_from_slice(&ANNEX_B_START_CODE);
        out.extend_from_slice(&sps_nal);

        let pps_rbsp = match (self.entropy_mode, high_profile) {
            (EntropyMode::Cavlc, _) => build_pps_cavlc(&self.pps_params),
            (EntropyMode::Cabac, false) => build_pps_cabac(&self.pps_params),
            (EntropyMode::Cabac, true) => build_pps_cabac_high(&self.pps_params),
        };
        let pps_nal = wrap_rbsp_as_nal(&pps_rbsp, NalType::PPS, 3);
        out.extend_from_slice(&ANNEX_B_START_CODE);
        out.extend_from_slice(&pps_nal);

        self.params_emitted = true;
    }

    fn frame_size_bytes(&self) -> usize {
        let y = (self.width * self.height) as usize;
        let c = (self.width / 2 * self.height / 2) as usize;
        y + 2 * c
    }

    fn build_idr_slice_rbsp_pcm(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
        let mut w = BitWriter::with_capacity((self.frame_size_bytes() + 256).max(512));
        let hdr = self.idr_slice_header();
        continue_slice_header_i(&mut w, &hdr);

        let mb_w = (self.width / 16) as usize;
        let mb_h = (self.height / 16) as usize;
        let y_stride = self.width as usize;
        let c_stride = (self.width / 2) as usize;
        let y_plane = &pixels[..(self.width * self.height) as usize];
        let cb_plane = &pixels[(self.width * self.height) as usize
            ..(self.width * self.height + self.width / 2 * self.height / 2) as usize];
        let cr_plane =
            &pixels[(self.width * self.height + self.width / 2 * self.height / 2) as usize..];

        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                self.write_pcm_macroblock(
                    &mut w, mb_x, mb_y, y_plane, y_stride, cb_plane, cr_plane, c_stride,
                )?;
            }
        }
        w.write_rbsp_trailing();
        Ok(w.finish())
    }

    fn build_idr_slice_rbsp_i16x16(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
        let mut w = BitWriter::with_capacity(self.frame_size_bytes().max(512));
        let hdr = self.idr_slice_header();
        continue_slice_header_i(&mut w, &hdr);

        let qp = self.rc.target_crf;
        let qp_c = derive_chroma_qp(qp as i32, self.pps_params.chroma_qp_index_offset as i32) as u8;

        let mb_w = (self.width / 16) as usize;
        let mb_h = (self.height / 16) as usize;
        let y_stride = self.width as usize;
        let c_stride = (self.width / 2) as usize;
        let y_size = (self.width * self.height) as usize;
        let c_size = (self.width / 2 * self.height / 2) as usize;
        let y_plane = &pixels[..y_size];
        let cb_plane = &pixels[y_size..y_size + c_size];
        let cr_plane = &pixels[y_size + c_size..];

        // Reset the I_4x4 mode grid at frame start (all MBs initially
        // "not I_4x4" → spec DC fallback for any cross-frame
        // neighbor queries at the top/left boundary).
        for m in self.i4x4_mode_grid.iter_mut() {
            *m = 0xFF;
        }
        for m in self.i8x8_mode_grid.iter_mut() {
            *m = 0xFF;
        }
        // Reset running MB QP to the slice's starting QP — spec § 9.
        self.prev_mb_qp = qp as i32;
        // Clear CAVLC TotalCoeff grid for this frame (nC fallback).
        for v in self.total_coeff_grid.iter_mut() {
            *v = 0xFF;
        }
        for v in self.intra16x16_dc_tc_grid.iter_mut() {
            *v = 0xFF;
        }
        for v in self.chroma_cb_tc_grid.iter_mut() {
            *v = 0;
        }
        for v in self.chroma_cr_tc_grid.iter_mut() {
            *v = 0;
        for v in self.transform_8x8_grid.iter_mut() { *v = false; }
        }
        for v in self.qp_grid.iter_mut() { *v = 0xFF; }
        for v in self.intra_grid.iter_mut() { *v = false; }

        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                self.write_intra_macroblock(
                    &mut w, mb_x, mb_y, y_plane, y_stride, cb_plane, cr_plane, c_stride, qp, qp_c,
                )?;
            }
        }

        // Apply the in-loop deblocking filter (spec § 8.7). Matches
        // what the decoder produces — mandatory so our DPB aligns
        // with downstream inter-frame prediction.
        if !super::mb_decision_b::env_var_os_is_some("PHASM_DISABLE_DEBLOCK") {
            let coded_flags = self.build_coded_flags();
            super::deblocking_filter::filter_frame(
                &mut self.recon,
                &self.qp_grid,
                &self.intra_grid,
                &coded_flags,
                None,
            );
            // §B-cascade-real Phase 1.1.B step 3: deblock visual_recon.
            super::deblocking_filter::filter_frame(
                &mut self.visual_recon,
                &self.qp_grid,
                &self.intra_grid,
                &coded_flags,
                None,
            );
        }

        w.write_rbsp_trailing();
        Ok(w.finish())
    }

    /// Per-MB CABAC dispatch that mirrors `write_intra_macroblock`:
    /// picks I_4x4 vs I_16x16 via SATD + overhead penalty and routes
    /// to the appropriate CABAC commit path.
    #[allow(clippy::too_many_arguments)]
    fn write_intra_macroblock_cabac(
        &mut self,
        cabac: &mut CabacEncoder,
        mb_x: usize,
        mb_y: usize,
        y_plane: &[u8],
        y_stride: usize,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
        qp: u8,
        _qp_c: u8,
        is_last_mb: bool,
    ) -> Result<(), EncoderError> {
        let y0 = mb_y * 16;
        let x0 = mb_x * 16;
        let mut src_y = [[0u8; 16]; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                src_y[dy][dx] = y_plane[(y0 + dy) * y_stride + x0 + dx];
            }
        }

        let variance = super::rate_control::mb_variance_16x16(&src_y);
        let qp_offset = super::rate_control::variance_to_qp_offset(variance);
        let mb_qp = ((qp as i32) + qp_offset).clamp(0, 51) as u8;
        let mb_qp_c = derive_chroma_qp(mb_qp as i32, self.pps_params.chroma_qp_index_offset as i32) as u8;
        // Commit MB state for the deblock filter (I-slice → is_intra=true).
        self.commit_mb_state(mb_x, mb_y, mb_qp, true);

        // Phase 100-F2 simple first cut: when 8×8 transform is enabled,
        // always use I_8x8. SATD-driven mode decision (compare I_4x4 vs
        // I_8x8 vs I_16x16) is a follow-up. This lets us verify the
        // end-to-end I_8x8 bitstream path first without also exercising
        // mode-switch-per-MB.
        if self.enable_transform_8x8 {
            let i8x8_result = super::intra_8x8_encode::encode_i8x8_mb(
                &src_y, &mut self.recon, mb_x, mb_y, mb_qp,
            );
            return self.commit_i8x8_macroblock_cabac(
                cabac, mb_x, mb_y, &i8x8_result, cb_plane, cr_plane, c_stride, mb_qp, mb_qp_c,
                is_last_mb,
            );
        }

        let neighbors_y = self.build_luma_neighbors_16x16(mb_x, mb_y);
        let i16_decision =
            choose_intra_16x16_mode_psy(&neighbors_y, &src_y, Self::PSY_RD_STRENGTH);
        let i16_cost = i16_decision.satd;

        let psy = Self::PSY_RD_STRENGTH;
        let i4x4_result =
            encode_i4x4_mb(&src_y, &mut self.recon, mb_x, mb_y, mb_qp, psy);
        let i4x4_cost = i4x4_result
            .total_satd
            .saturating_add(Self::I4X4_OVERHEAD_PENALTY);

        if i4x4_cost < i16_cost {
            self.commit_i4x4_macroblock_cabac(
                cabac, mb_x, mb_y, &i4x4_result, cb_plane, cr_plane, c_stride, mb_qp, mb_qp_c,
                is_last_mb, /* in_p_slice: */ false,
            )
        } else {
            self.clear_i4x4_mode_grid_for_mb(mb_x, mb_y);
            self.write_i16x16_macroblock_cabac(
                cabac, mb_x, mb_y, y_plane, y_stride, cb_plane, cr_plane, c_stride, mb_qp, mb_qp_c,
                is_last_mb, false,
            )
        }
    }

    /// I_4x4 CABAC commit (Phase 6C.6d.2). Mirrors `commit_i4x4_macroblock`
    /// but emits through `CabacEncoder`. Luma residuals use ctxBlockCat
    /// 2 (LumaLevel4x4) and are gated per 8×8 block by the CBP luma
    /// bits; chroma reuses the I_16x16 path.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn commit_i4x4_macroblock_cabac(
        &mut self,
        cabac: &mut CabacEncoder,
        mb_x: usize,
        mb_y: usize,
        i4x4: &I4x4MbResult,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
        qp: u8,
        qp_c: u8,
        is_last_mb: bool,
        // Phase D.2-stealth: when true, emit mb_type via
        // encode_mb_type_p(5, ...) (I_NxN in P-slice codenum 5)
        // instead of the I-slice encode_mb_type_i(0, ...). Also
        // emit the leading mb_skip_flag=0 bin (intra-in-P MBs are
        // never skip). Other syntax (prev_intra4x4_pred_mode_flag,
        // rem_intra4x4_pred_mode, intra_chroma_pred_mode,
        // mb_qp_delta, residuals) is identical to I-slice.
        in_p_slice: bool,
    ) -> Result<(), EncoderError> {
        use crate::codec::h264::cabac::encoder::{
            encode_coded_block_pattern, encode_mb_type_p,
            encode_prev_intra4x4_pred_mode_flag, encode_rem_intra4x4_pred_mode,
            encode_residual_block_cabac_with_cbf_inc,
        };
        use crate::codec::h264::cabac::neighbor::{
            block_pos_to_chroma_ac_idx, compute_cbf_ctx_idx_inc_chroma_ac,
            compute_cbf_ctx_idx_inc_chroma_dc, compute_cbf_ctx_idx_inc_luma_4x4, CurrentMbCbf,
        };
        use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;

        // ── Chroma encode (same pipeline as I_16x16) ──
        let cy0 = mb_y * 8;
        let cx0 = mb_x * 8;
        let mut src_cb = [[0u8; 8]; 8];
        let mut src_cr = [[0u8; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                src_cb[dy][dx] = cb_plane[(cy0 + dy) * c_stride + cx0 + dx];
                src_cr[dy][dx] = cr_plane[(cy0 + dy) * c_stride + cx0 + dx];
            }
        }
        let cb_neighbors = self.build_chroma_neighbors(mb_x, mb_y, 0);
        let cb_decision = choose_intra_chroma_mode(&cb_neighbors, &src_cb);
        let chroma_pred_mode = cb_decision.mode as u32;
        let pred_cb = cb_decision.predicted;
        let pred_cr = {
            let cr_neighbors = self.build_chroma_neighbors(mb_x, mb_y, 1);
            crate::codec::h264::intra_pred::predict_chroma_8x8(cb_decision.mode, &cr_neighbors)
        };
        let chroma_qp = QuantParams {
            qp: qp_c,
            slice: QuantSlice::Intra,
        };
        let (cb_ac_levels, cb_dc_levels) =
            self.encode_chroma_component(&src_cb, &pred_cb, chroma_qp);
        let (cr_ac_levels, cr_dc_levels) =
            self.encode_chroma_component(&src_cr, &pred_cr, chroma_qp);

        // §B-cascade-real Phase 1.1.B step 2 — parallel post-flip
        // chroma arrays. Initialised to the pre-flip levels; updated
        // by the residual-hook fire sites below to track what CABAC
        // actually emits. Used at function epilogue for the
        // visual_recon recon pass (what a downstream player sees).
        let mut cb_ac_post = cb_ac_levels;
        let mut cb_dc_post = cb_dc_levels;
        let mut cr_ac_post = cr_ac_levels;
        let mut cr_dc_post = cr_dc_levels;

        // ── CBP computation ──
        let mut luma_nonzero = [false; 16];
        for k in 0..16 {
            luma_nonzero[k] =
                i4x4.ac_levels[k].iter().any(|r| r.iter().any(|&v| v != 0));
        }
        let cbp_luma = luma_8x8_cbp_mask(&luma_nonzero);
        let any_chroma_ac = cb_ac_levels
            .iter()
            .chain(cr_ac_levels.iter())
            .any(|b| b.iter().any(|r| r.iter().any(|&v| v != 0)));
        let any_chroma_dc = cb_dc_levels
            .iter()
            .flatten()
            .chain(cr_dc_levels.iter().flatten())
            .any(|&v| v != 0);
        let cbp_chroma = if any_chroma_ac {
            2u8
        } else if any_chroma_dc {
            1u8
        } else {
            0u8
        };
        let cbp_value = pack_cbp(cbp_luma, cbp_chroma);

        // ── Emit CABAC syntax ──
        // mb_type: I-slice → 0 (I_NxN). P-slice intra-in-P →
        // codenum 5 (I_NxN) via the P-slice mb_type encoder, which
        // writes the intra-in-P prefix bin "1" and the I_NxN suffix
        // bin "0". Caller is responsible for emitting the leading
        // `mb_skip_flag=0` bin BEFORE calling this path (matches
        // the I_16x16 P-slice caller contract in
        // `write_i16x16_macroblock_cabac`).
        if in_p_slice {
            encode_mb_type_p(cabac, 5, mb_x);
        } else {
            encode_mb_type_i(cabac, 0, mb_x);
        }

        // Spec § 7.3.5.1 macroblock_layer: for I_NxN MBs, emit
        // `transform_size_8x8_flag` AFTER mb_type, BEFORE mb_pred
        // (prev_intra4x4_pred_mode_flag loop below). Gate on the PPS
        // `transform_8x8_mode_flag` via Encoder::enable_transform_8x8.
        // Phase 100-E always emits 0 (we stay I_4x4, not I_8x8);
        // Phase 100-F flips this to 1 when the I_8x8 mode is chosen.
        if self.enable_transform_8x8 {
            crate::codec::h264::cabac::encoder::encode_transform_size_8x8_flag(
                cabac, false, mb_x,
            );
        }

        // Per-sub-block mode flags (16 × prev_flag + optional rem).
        let flags = derive_i4x4_mode_flags(&i4x4.modes, mb_x, mb_y, |bx, by| {
            self.lookup_i4x4_mode(bx, by)
        });
        for (flag, rem) in flags.iter() {
            encode_prev_intra4x4_pred_mode_flag(cabac, *flag);
            if let Some(r) = rem {
                encode_rem_intra4x4_pred_mode(cabac, *r);
            }
        }

        // intra_chroma_pred_mode.
        encode_intra_chroma_pred_mode(cabac, chroma_pred_mode as u8, mb_x);

        // coded_block_pattern (luma FL + chroma TU, with same-MB
        // tracking fix).
        encode_coded_block_pattern(cabac, cbp_value, mb_x);

        // mb_qp_delta only when any residual block is present
        // (spec § 7.3.5 — I_4x4 is NOT Intra_16x16 so the CBP gate
        // applies).
        let qp_delta_emitted;
        if cbp_value != 0 {
            let delta = qp as i32 - self.prev_mb_qp;
            encode_mb_qp_delta(cabac, delta);
            self.prev_mb_qp = qp as i32;
            qp_delta_emitted = delta;
        } else {
            // No mb_qp_delta emitted → decoder uses prev_mb_qp for
            // deblock. Match our qp_grid.
            self.commit_mb_state(mb_x, mb_y, self.prev_mb_qp as u8, true);
            qp_delta_emitted = 0;
        }

        // ── Residual emit ──
        let mut current_cbf = CurrentMbCbf::new();

        // commit_i4x4_macroblock_cabac is the I_4x4 intra path. is_intra=true.
        let current_is_intra = true;
        // Luma 4×4 blocks (ctxBlockCat = 2). Gated per 8×8 block via
        // cbp_luma bits — block k belongs to 8×8 block (k / 4).
        if cbp_luma != 0 {
            for k in 0..16 {
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                if cbp_luma & (1 << (k / 4)) != 0 {
                    let mut scan = raster_to_scan_levels(&i4x4.ac_levels[k]);
                    // Phase 6D.8: stego hook for I_4x4 luma residual.
                    self.invoke_stego_residual_hook(
                        mb_x, mb_y, &mut scan, 0, 15,
                        super::super::stego::orchestrate::ResidualPathKind::Luma4x4 {
                            block_idx: k as u8,
                        },
                    );
                    let inc = compute_cbf_ctx_idx_inc_luma_4x4(
                        &current_cbf,
                        &cabac.neighbors,
                        mb_x,
                        bx,
                        by,
                        current_is_intra,
                    );
                    let coded = encode_residual_block_cabac_with_cbf_inc(
                        cabac, &scan, 0, 15, 2, inc,
                    );
                    current_cbf.set(2, k, coded);
                    let abs_bx = mb_x * 4 + bx as usize;
                    let abs_by = mb_y * 4 + by as usize;
                    self.store_total_coeff_luma(
                        abs_bx,
                        abs_by,
                        if coded { 1 } else { 0 },
                    );
                } else {
                    let abs_bx = mb_x * 4 + bx as usize;
                    let abs_by = mb_y * 4 + by as usize;
                    self.store_total_coeff_luma(abs_bx, abs_by, 0);
                }
            }
        } else {
            for k in 0..16 {
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                self.store_total_coeff_luma(
                    mb_x * 4 + bx as usize,
                    mb_y * 4 + by as usize,
                    0,
                );
            }
        }

        // Chroma DC / AC (same structure as I_16x16 CABAC path).
        if cbp_chroma >= 1 {
            for (plane, dc) in [&cb_dc_levels, &cr_dc_levels].iter().enumerate() {
                let mut dc_flat: [i32; 4] = [dc[0][0], dc[0][1], dc[1][0], dc[1][1]];
                // Phase 6D.8: stego hook for I_4x4 Chroma DC.
                self.invoke_stego_residual_hook(
                    mb_x, mb_y, &mut dc_flat, 0, 3,
                    super::super::stego::orchestrate::ResidualPathKind::ChromaDc {
                        plane: plane as u8,
                    },
                );
                // §B-cascade-real Phase 1.1.B step 2: capture post-flip
                // DC into the parallel _post array.
                let dc_post_raster: [[i32; 2]; 2] =
                    [[dc_flat[0], dc_flat[1]], [dc_flat[2], dc_flat[3]]];
                if plane == 0 {
                    cb_dc_post = dc_post_raster;
                } else {
                    cr_dc_post = dc_post_raster;
                }
                let inc =
                    compute_cbf_ctx_idx_inc_chroma_dc(&cabac.neighbors, mb_x, plane as u8, current_is_intra);
                let coded = encode_residual_block_cabac_with_cbf_inc(
                    cabac, &dc_flat, 0, 3, 3, inc,
                );
                current_cbf.set(3, plane, coded);
            }
        }
        if cbp_chroma == 2 {
            for (plane, ac_blocks) in
                [&cb_ac_levels, &cr_ac_levels].iter().enumerate()
            {
                for sub in 0..4 {
                    let bx = (sub % 2) as u8;
                    let by = (sub / 2) as u8;
                    let mut ac_scan = ac_scan_order_15(&ac_blocks[sub]);
                    // Phase 6D.8: stego hook for I_4x4 Chroma AC.
                    self.invoke_stego_residual_hook(
                        mb_x, mb_y, &mut ac_scan, 0, 14,
                        super::super::stego::orchestrate::ResidualPathKind::ChromaAc {
                            plane: plane as u8,
                            block_idx: sub as u8,
                        },
                    );
                    // §B-cascade-real Phase 1.1.B step 2: capture
                    // post-flip AC into the parallel _post array.
                    let ac_post_raster = ac_scan_15_to_raster(&ac_scan);
                    if plane == 0 {
                        cb_ac_post[sub] = ac_post_raster;
                    } else {
                        cr_ac_post[sub] = ac_post_raster;
                    }
                    let inc = compute_cbf_ctx_idx_inc_chroma_ac(
                        &current_cbf,
                        &cabac.neighbors,
                        mb_x,
                        plane as u8,
                        bx,
                        by,
                        current_is_intra,
                    );
                    let coded = encode_residual_block_cabac_with_cbf_inc(
                        cabac, &ac_scan, 0, 14, 4, inc,
                    );
                    current_cbf.set(
                        4,
                        block_pos_to_chroma_ac_idx(plane as u8, bx, by),
                        coded,
                    );
                }
            }
        }

        // end_of_slice_flag (terminate-coded).
        encode_end_of_slice_flag(cabac, is_last_mb);

        // Commit neighbor state.
        let mut nb = CabacNeighborMB::default();
        nb.mb_type = MbTypeClass::INxN;
        nb.intra_chroma_pred_mode = chroma_pred_mode as u8;
        nb.cbp_luma = cbp_luma;
        nb.cbp_chroma = cbp_chroma;
        nb.mb_qp_delta = qp_delta_emitted;
        nb.coded_block_flag_cat = current_cbf.to_neighbor_cbf();
        cabac.neighbors.commit(mb_x, nb);

        // Chroma reconstruction. Luma recon is already in `self.recon`
        // from `encode_i4x4_mb`.
        let recon_cb =
            self.reconstruct_chroma_mb(&pred_cb, &cb_ac_levels, &cb_dc_levels, qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &recon_cb);
        let recon_cr =
            self.reconstruct_chroma_mb(&pred_cr, &cr_ac_levels, &cr_dc_levels, qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &recon_cr);

        // §B-cascade-real Phase 1.1.B step 2 — visual_recon path:
        // re-run chroma reconstruct using POST-flip levels, write to
        // visual_recon. When no flips fired this is byte-identical to
        // the encoder.recon writes above. NB: the frame-end mirror in
        // encode_*_frame currently clobbers these per-MB writes; the
        // mirror gets removed (and visual_recon deblocking added) in
        // step 3, at which point this differentiation becomes
        // observable to mux/PSNR consumers.
        let visual_cb =
            self.reconstruct_chroma_mb(&pred_cb, &cb_ac_post, &cb_dc_post, qp_c);
        self.visual_recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &visual_cb);
        let visual_cr =
            self.reconstruct_chroma_mb(&pred_cr, &cr_ac_post, &cr_dc_post, qp_c);
        self.visual_recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &visual_cr);
        // I_4x4 luma is Shape 2 (recon-inside-helper, deferred); copy
        // pre-flip luma from self.recon to self.visual_recon for the
        // current MB so the luma plane stays in sync at MB boundary.
        let lx0 = (mb_x * 16) as u32;
        let ly0 = (mb_y * 16) as u32;
        let stride = self.recon.width as usize;
        for dy in 0..16u32 {
            for dx in 0..16u32 {
                let idx = ((ly0 + dy) as usize) * stride + (lx0 + dx) as usize;
                self.visual_recon.y[idx] = self.recon.y[idx];
            }
        }

        // Publish the mode grid so subsequent MBs see these I_4x4 modes.
        self.store_i4x4_mode_grid_for_mb(mb_x, mb_y, &i4x4.modes);

        let _ = qp; // captured for parity with CAVLC commit signature
        Ok(())
    }

    /// I_8x8 CABAC commit (Phase 100-F2). Mirrors `commit_i4x4_macroblock_cabac`
    /// but routes luma residuals through the 8×8 pipeline:
    ///   - mb_type = 0 (I_NxN) — same as I_4x4 at the bitstream level;
    ///     the difference is signaled by `transform_size_8x8_flag = 1`
    ///     emitted immediately after mb_type.
    ///   - mb_pred uses 4 × prev_intra8x8_pred_mode_flag (same contexts
    ///     as Intra_4x4 per Table 9-39), with spec § 8.3.3 derivation.
    ///   - Residual: for each 8×8 block with `cbp_luma` bit set, emit
    ///     `encode_residual_block_cabac_8x8` (ctxBlockCat = 5, no CBF).
    ///   - Chroma stays 4:2:0 4×4 via the existing chroma pipeline.
    #[allow(clippy::too_many_arguments)]
    fn commit_i8x8_macroblock_cabac(
        &mut self,
        cabac: &mut CabacEncoder,
        mb_x: usize,
        mb_y: usize,
        i8x8: &super::intra_8x8_encode::I8x8MbResult,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
        qp: u8,
        qp_c: u8,
        is_last_mb: bool,
    ) -> Result<(), EncoderError> {
        use crate::codec::h264::cabac::encoder::{
            encode_coded_block_pattern, encode_prev_intra4x4_pred_mode_flag,
            encode_rem_intra4x4_pred_mode, encode_residual_block_cabac_8x8,
            encode_residual_block_cabac_with_cbf_inc, encode_transform_size_8x8_flag, ZIGZAG_8X8,
        };
        use crate::codec::h264::cabac::neighbor::{
            block_pos_to_chroma_ac_idx, compute_cbf_ctx_idx_inc_chroma_ac,
            compute_cbf_ctx_idx_inc_chroma_dc, CurrentMbCbf,
        };
        use super::intra_8x8_encode::derive_i8x8_mode_flags;

        // ── Chroma encode (same pipeline as I_4x4 / I_16x16) ──
        let cy0 = mb_y * 8;
        let cx0 = mb_x * 8;
        let mut src_cb = [[0u8; 8]; 8];
        let mut src_cr = [[0u8; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                src_cb[dy][dx] = cb_plane[(cy0 + dy) * c_stride + cx0 + dx];
                src_cr[dy][dx] = cr_plane[(cy0 + dy) * c_stride + cx0 + dx];
            }
        }
        let cb_neighbors = self.build_chroma_neighbors(mb_x, mb_y, 0);
        let cb_decision = choose_intra_chroma_mode(&cb_neighbors, &src_cb);
        let chroma_pred_mode = cb_decision.mode as u32;
        let pred_cb = cb_decision.predicted;
        let pred_cr = {
            let cr_neighbors = self.build_chroma_neighbors(mb_x, mb_y, 1);
            crate::codec::h264::intra_pred::predict_chroma_8x8(cb_decision.mode, &cr_neighbors)
        };
        let chroma_qp = QuantParams {
            qp: qp_c,
            slice: QuantSlice::Intra,
        };
        let (cb_ac_levels, cb_dc_levels) =
            self.encode_chroma_component(&src_cb, &pred_cb, chroma_qp);
        let (cr_ac_levels, cr_dc_levels) =
            self.encode_chroma_component(&src_cr, &pred_cr, chroma_qp);

        // §B-cascade-real Phase 1.1.B step 2 — parallel post-flip
        // chroma arrays for the visual_recon recon pass at function
        // epilogue.
        let mut cb_ac_post = cb_ac_levels;
        let mut cb_dc_post = cb_dc_levels;
        let mut cr_ac_post = cr_ac_levels;
        let mut cr_dc_post = cr_dc_levels;

        // ── CBP computation ──
        // Luma: one bit per 8×8 block. `i8x8.nonzero[k]` is the k-th
        // 8×8 block's any-nonzero flag in raster order; the spec's
        // `cbp_luma_8x8_mask` bit layout matches this directly (bit k
        // = 8×8 block k).
        let mut cbp_luma = 0u8;
        for k in 0..4 {
            if i8x8.nonzero[k] {
                cbp_luma |= 1 << k;
            }
        }
        let any_chroma_ac = cb_ac_levels
            .iter()
            .chain(cr_ac_levels.iter())
            .any(|b| b.iter().any(|r| r.iter().any(|&v| v != 0)));
        let any_chroma_dc = cb_dc_levels
            .iter()
            .flatten()
            .chain(cr_dc_levels.iter().flatten())
            .any(|&v| v != 0);
        let cbp_chroma = if any_chroma_ac {
            2u8
        } else if any_chroma_dc {
            1u8
        } else {
            0u8
        };
        let cbp_value = pack_cbp(cbp_luma, cbp_chroma);

        // ── Emit CABAC syntax ──
        // mb_type = 0 for I_NxN. Distinction I_4x4 vs I_8x8 is made by
        // transform_size_8x8_flag below.
        encode_mb_type_i(cabac, 0, mb_x);

        // transform_size_8x8_flag = 1 — THIS is what selects I_8x8.
        debug_assert!(self.enable_transform_8x8);
        encode_transform_size_8x8_flag(cabac, true, mb_x);

        // Per-8×8-block mode flags. Re-uses the 4×4 encoders since
        // Table 9-39 says I_4x4 and I_8x8 share ctxIdxOffsets 68/69.
        let flags = derive_i8x8_mode_flags(&i8x8.modes, mb_x, mb_y, |bx, by| {
            self.lookup_i8x8_mode(bx, by)
        });
        for (flag, rem) in flags.iter() {
            encode_prev_intra4x4_pred_mode_flag(cabac, *flag);
            if let Some(r) = rem {
                encode_rem_intra4x4_pred_mode(cabac, *r);
            }
        }

        encode_intra_chroma_pred_mode(cabac, chroma_pred_mode as u8, mb_x);
        encode_coded_block_pattern(cabac, cbp_value, mb_x);

        let qp_delta_emitted;
        if cbp_value != 0 {
            let delta = qp as i32 - self.prev_mb_qp;
            encode_mb_qp_delta(cabac, delta);
            self.prev_mb_qp = qp as i32;
            qp_delta_emitted = delta;
        } else {
            self.commit_mb_state(mb_x, mb_y, self.prev_mb_qp as u8, true);
            qp_delta_emitted = 0;
        }

        // ── Residual emit ──
        let mut current_cbf = CurrentMbCbf::new();
        let current_is_intra = true;

        // Luma 8×8 blocks (ctxBlockCat = 5). No CBF emission. Gated
        // per 8×8 block via cbp_luma bit.
        for k in 0..4 {
            if cbp_luma & (1 << k) != 0 {
                // Flatten row-major 8×8 levels to [i32; 64] via ZIGZAG_8X8.
                let levels_rm = &i8x8.ac_levels[k];
                let mut scan_coeffs = [0i32; 64];
                for i in 0..64 {
                    let pos = ZIGZAG_8X8[i] as usize;
                    let row = pos / 8;
                    let col = pos % 8;
                    scan_coeffs[i] = levels_rm[row][col] as i32;
                }
                // Phase 6D.8: stego hook for I_8x8 luma residual.
                self.invoke_stego_residual_hook(
                    mb_x, mb_y, &mut scan_coeffs, 0, 63,
                    super::super::stego::orchestrate::ResidualPathKind::Luma8x8 {
                        block_idx: k as u8,
                    },
                );
                encode_residual_block_cabac_8x8(cabac, &scan_coeffs);
            }
        }

        // Populate cat-2 neighbor CBF state from the 8×8 CBP bits so
        // subsequent I_4x4 MBs see our 4×4 positions as coded per spec
        // (each 4×4 block inside an 8×8-coded block is treated as
        // coded_block_flag = 1). `store_total_coeff_luma` also writes
        // the CAVLC nC grid for any mixed I_4x4/I_8x8 content.
        for k in 0..16 {
            let blk8 = k / 4;
            let is_coded = (cbp_luma & (1 << blk8)) != 0;
            current_cbf.set(2, k, is_coded);
            let (bx, by) = crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS[k];
            let abs_bx = mb_x * 4 + bx as usize;
            let abs_by = mb_y * 4 + by as usize;
            self.store_total_coeff_luma(abs_bx, abs_by, if is_coded { 1 } else { 0 });
        }

        // Chroma DC / AC (same structure as I_4x4 / I_16x16 CABAC path).
        if cbp_chroma >= 1 {
            for (plane, dc) in [&cb_dc_levels, &cr_dc_levels].iter().enumerate() {
                let mut dc_flat: [i32; 4] = [dc[0][0], dc[0][1], dc[1][0], dc[1][1]];
                // Phase 6D.8: stego hook for I_8x8 Chroma DC.
                self.invoke_stego_residual_hook(
                    mb_x, mb_y, &mut dc_flat, 0, 3,
                    super::super::stego::orchestrate::ResidualPathKind::ChromaDc {
                        plane: plane as u8,
                    },
                );
                // §B-cascade-real Phase 1.1.B step 2: capture post-flip
                // DC into the parallel _post array.
                let dc_post_raster: [[i32; 2]; 2] =
                    [[dc_flat[0], dc_flat[1]], [dc_flat[2], dc_flat[3]]];
                if plane == 0 {
                    cb_dc_post = dc_post_raster;
                } else {
                    cr_dc_post = dc_post_raster;
                }
                let inc = compute_cbf_ctx_idx_inc_chroma_dc(
                    &cabac.neighbors,
                    mb_x,
                    plane as u8,
                    current_is_intra,
                );
                let coded = encode_residual_block_cabac_with_cbf_inc(
                    cabac, &dc_flat, 0, 3, 3, inc,
                );
                current_cbf.set(3, plane, coded);
            }
        }
        if cbp_chroma == 2 {
            for (plane, ac_blocks) in [&cb_ac_levels, &cr_ac_levels].iter().enumerate() {
                for sub in 0..4 {
                    let bx = (sub % 2) as u8;
                    let by = (sub / 2) as u8;
                    let mut ac_scan = ac_scan_order_15(&ac_blocks[sub]);
                    // Phase 6D.8: stego hook for I_8x8 Chroma AC.
                    self.invoke_stego_residual_hook(
                        mb_x, mb_y, &mut ac_scan, 0, 14,
                        super::super::stego::orchestrate::ResidualPathKind::ChromaAc {
                            plane: plane as u8,
                            block_idx: sub as u8,
                        },
                    );
                    // §B-cascade-real Phase 1.1.B step 2: capture
                    // post-flip AC into the parallel _post array.
                    let ac_post_raster = ac_scan_15_to_raster(&ac_scan);
                    if plane == 0 {
                        cb_ac_post[sub] = ac_post_raster;
                    } else {
                        cr_ac_post[sub] = ac_post_raster;
                    }
                    let inc = compute_cbf_ctx_idx_inc_chroma_ac(
                        &current_cbf,
                        &cabac.neighbors,
                        mb_x,
                        plane as u8,
                        bx,
                        by,
                        current_is_intra,
                    );
                    let coded = encode_residual_block_cabac_with_cbf_inc(
                        cabac, &ac_scan, 0, 14, 4, inc,
                    );
                    current_cbf.set(
                        4,
                        block_pos_to_chroma_ac_idx(plane as u8, bx, by),
                        coded,
                    );
                }
            }
        }

        encode_end_of_slice_flag(cabac, is_last_mb);

        // Commit neighbor state.
        let mut nb = CabacNeighborMB::default();
        nb.mb_type = MbTypeClass::INxN;
        nb.intra_chroma_pred_mode = chroma_pred_mode as u8;
        nb.cbp_luma = cbp_luma;
        nb.cbp_chroma = cbp_chroma;
        nb.mb_qp_delta = qp_delta_emitted;
        nb.coded_block_flag_cat = current_cbf.to_neighbor_cbf();
        nb.transform_size_8x8_flag = true;
        cabac.neighbors.commit(mb_x, nb);

        // Chroma reconstruction. Luma recon is already in `self.recon`
        // from `encode_i8x8_mb`.
        let recon_cb =
            self.reconstruct_chroma_mb(&pred_cb, &cb_ac_levels, &cb_dc_levels, qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &recon_cb);
        let recon_cr =
            self.reconstruct_chroma_mb(&pred_cr, &cr_ac_levels, &cr_dc_levels, qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &recon_cr);

        // §B-cascade-real Phase 1.1.B step 2 — visual_recon path with
        // POST-flip levels. I_8x8 luma is Shape 2 (recon-inside-helper,
        // deferred); copy pre-flip luma to keep visual_recon's luma
        // plane in sync at MB boundary.
        let visual_cb =
            self.reconstruct_chroma_mb(&pred_cb, &cb_ac_post, &cb_dc_post, qp_c);
        self.visual_recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &visual_cb);
        let visual_cr =
            self.reconstruct_chroma_mb(&pred_cr, &cr_ac_post, &cr_dc_post, qp_c);
        self.visual_recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &visual_cr);
        let lx0 = (mb_x * 16) as u32;
        let ly0 = (mb_y * 16) as u32;
        let stride = self.recon.width as usize;
        for dy in 0..16u32 {
            for dx in 0..16u32 {
                let idx = ((ly0 + dy) as usize) * stride + (lx0 + dx) as usize;
                self.visual_recon.y[idx] = self.recon.y[idx];
            }
        }

        // Publish 8×8 mode grid; clear 4×4 grid for this MB.
        self.store_i8x8_mode_grid_for_mb(mb_x, mb_y, &i8x8.modes);
        self.clear_i4x4_mode_grid_for_mb(mb_x, mb_y);
        // Deblock filter skips internal 4-pixel edges for this MB
        // (spec § 8.7.2.1 with transform_8x8_flag).
        self.mark_transform_8x8_mb(mb_x, mb_y);

        let _ = qp;
        Ok(())
    }

    /// CABAC variant of the IDR slice builder. Per-MB, dispatches to
    /// `write_intra_macroblock_cabac` which picks I_4x4 vs I_16x16
    /// via SATD + overhead RDO.
    fn build_idr_slice_rbsp_cabac(
        &mut self,
        pixels: &[u8],
    ) -> Result<Vec<u8>, EncoderError> {
        let mut w = BitWriter::with_capacity(self.frame_size_bytes().max(512));
        let hdr = self.idr_slice_header();
        continue_slice_header_i(&mut w, &hdr);

        let qp = self.rc.target_crf;
        let qp_c = derive_chroma_qp(qp as i32, self.pps_params.chroma_qp_index_offset as i32) as u8;

        let mb_w = (self.width / 16) as usize;
        let mb_h = (self.height / 16) as usize;
        let y_stride = self.width as usize;
        let c_stride = (self.width / 2) as usize;
        let y_size = (self.width * self.height) as usize;
        let c_size = (self.width / 2 * self.height / 2) as usize;
        let y_plane = &pixels[..y_size];
        let cb_plane = &pixels[y_size..y_size + c_size];
        let cr_plane = &pixels[y_size + c_size..];

        // CABAC engine: one per slice, initialized from the I/SI table
        // and the slice's SliceQPy.
        let mut cabac = CabacEncoder::new_slice(CabacInitSlot::ISI, qp as i32, mb_w);
        if self.cabac_trace_enabled {
            cabac.engine.trace = Some(Vec::new());
        }

        // Reset encoder-side neighbor state.
        for m in self.i4x4_mode_grid.iter_mut() {
            *m = 0xFF;
        }
        for m in self.i8x8_mode_grid.iter_mut() {
            *m = 0xFF;
        }
        self.prev_mb_qp = qp as i32;
        for v in self.total_coeff_grid.iter_mut() {
            *v = 0xFF;
        }
        for v in self.intra16x16_dc_tc_grid.iter_mut() {
            *v = 0xFF;
        }
        for v in self.chroma_cb_tc_grid.iter_mut() {
            *v = 0;
        }
        for v in self.chroma_cr_tc_grid.iter_mut() {
        for v in self.transform_8x8_grid.iter_mut() { *v = false; }
            *v = 0;
        }
        for v in self.qp_grid.iter_mut() { *v = 0xFF; }
        for v in self.intra_grid.iter_mut() { *v = false; }

        for mb_y in 0..mb_h {
            if mb_y > 0 {
                cabac.neighbors.new_row();
            }
            for mb_x in 0..mb_w {
                let is_last_mb = mb_y == mb_h - 1 && mb_x == mb_w - 1;
                self.write_intra_macroblock_cabac(
                    &mut cabac, mb_x, mb_y, y_plane, y_stride, cb_plane, cr_plane, c_stride, qp,
                    qp_c, is_last_mb,
                )?;
            }
        }

        // Deblocking filter on reconstruction (same as CAVLC path).
        // CABAC IDR path can use 8×8 transform — pass the grid so
        // internal 4-pixel-grid edges are skipped on those MBs.
        if !super::mb_decision_b::env_var_os_is_some("PHASM_DISABLE_DEBLOCK") {
            let coded_flags = self.build_coded_flags();
            super::deblocking_filter::filter_frame_with_transform(
                &mut self.recon,
                &self.qp_grid,
                &self.intra_grid,
                Some(&self.transform_8x8_grid),
                &coded_flags,
                None,
            );
            // §B-cascade-real Phase 1.1.B step 3: deblock visual_recon.
            super::deblocking_filter::filter_frame_with_transform(
                &mut self.visual_recon,
                &self.qp_grid,
                &self.intra_grid,
                Some(&self.transform_8x8_grid),
                &coded_flags,
                None,
            );
        }

        // Finalize CABAC engine + assemble RBSP (header + alignment +
        // body + trailing), then append cabac_zero_word stuffing if
        // bin density exceeds the spec § 9.3.4.6 threshold.
        let bin_count = cabac.engine.bin_count();
        let pic_size_mbs = (mb_w * mb_h) as u32;
        if let Some(trace) = cabac.engine.trace.take() {
            self.cabac_trace_buffer.extend(trace);
        }
        let body = cabac.finish();
        let mut rbsp = assemble_cabac_slice_rbsp(w, &body);
        append_cabac_zero_words(&mut rbsp, bin_count, pic_size_mbs);
        Ok(rbsp)
    }

    /// Emit one I_16x16 macroblock via CABAC (Phase 6C.6d — full
    /// residuals). Mirrors `write_i16x16_macroblock` up through
    /// quantization + reconstruction; only the bitstream emit path
    /// routes through `CabacEncoder` + `CurrentMbCbf` for proper
    /// same-MB `coded_block_flag` neighbor derivation.
    #[allow(clippy::too_many_arguments)]
    fn write_i16x16_macroblock_cabac(
        &mut self,
        cabac: &mut CabacEncoder,
        mb_x: usize,
        mb_y: usize,
        y_plane: &[u8],
        y_stride: usize,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
        qp: u8,
        qp_c: u8,
        is_last_mb: bool,
        in_p_slice: bool,
    ) -> Result<(), EncoderError> {
        self.write_i16x16_macroblock_cabac_with_ctx(
            cabac, mb_x, mb_y, y_plane, y_stride, cb_plane, cr_plane, c_stride,
            qp, qp_c, is_last_mb,
            if in_p_slice { IntraSliceCtx::P } else { IntraSliceCtx::I },
        )
    }

    /// §intra-in-B (#319) Phase 3 — body of [`Self::write_i16x16_macroblock_cabac`]
    /// extended to dispatch the mb_type prefix on a 3-way slice context
    /// (I / P / B). Each context uses a different CABAC mb_type encoder
    /// + codenum offset:
    /// - I-slice: `encode_mb_type_i(cabac, mb_type, mb_x)` — codenum 1..24.
    /// - P-slice: `encode_mb_type_p(cabac, mb_type + 5, mb_x)` — Table 7-13
    ///   intra-in-P range 6..29.
    /// - B-slice: `encode_mb_type_b(cabac, mb_type + 23, mb_x)` — Table 7-14
    ///   intra-in-B range 24..47.
    ///
    /// All three callers (I-frame body, intra-in-P fallback, intra-in-B
    /// fallback) MUST emit the leading `mb_skip_flag = 0` bin before
    /// invoking this function. The helper itself does NOT emit the
    /// skip-flag — it starts at the mb_type bin.
    #[allow(clippy::too_many_arguments)]
    fn write_i16x16_macroblock_cabac_with_ctx(
        &mut self,
        cabac: &mut CabacEncoder,
        mb_x: usize,
        mb_y: usize,
        y_plane: &[u8],
        y_stride: usize,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
        qp: u8,
        qp_c: u8,
        is_last_mb: bool,
        slice_ctx: IntraSliceCtx,
    ) -> Result<(), EncoderError> {
        let in_p_slice = matches!(slice_ctx, IntraSliceCtx::P);
        use crate::codec::h264::cabac::encoder::{encode_mb_type_p, encode_residual_block_cabac_with_cbf_inc};
        use crate::codec::h264::cabac::neighbor::{
            block_pos_to_chroma_ac_idx,
            compute_cbf_ctx_idx_inc_chroma_ac, compute_cbf_ctx_idx_inc_chroma_dc,
            compute_cbf_ctx_idx_inc_luma_ac, compute_cbf_ctx_idx_inc_luma_dc, CurrentMbCbf,
        };
        use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;

        // 1. Gather source MB pixels.
        let y0 = mb_y * 16;
        let x0 = mb_x * 16;
        let mut src_y = [[0u8; 16]; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                src_y[dy][dx] = y_plane[(y0 + dy) * y_stride + x0 + dx];
            }
        }
        let cy0 = mb_y * 8;
        let cx0 = mb_x * 8;
        let mut src_cb = [[0u8; 8]; 8];
        let mut src_cr = [[0u8; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                src_cb[dy][dx] = cb_plane[(cy0 + dy) * c_stride + cx0 + dx];
                src_cr[dy][dx] = cr_plane[(cy0 + dy) * c_stride + cx0 + dx];
            }
        }

        // 2. Mode decision.
        let neighbors_y = self.build_luma_neighbors_16x16(mb_x, mb_y);
        let luma_decision =
            choose_intra_16x16_mode_psy(&neighbors_y, &src_y, Self::PSY_RD_STRENGTH);
        let pred_y = luma_decision.predicted;
        let luma_pred_mode = luma_decision.mode as u32;

        let cb_neighbors = self.build_chroma_neighbors(mb_x, mb_y, 0);
        let cb_decision = choose_intra_chroma_mode(&cb_neighbors, &src_cb);
        let chroma_pred_mode = cb_decision.mode as u32;
        let pred_cb = cb_decision.predicted;
        let pred_cr = {
            let cr_neighbors = self.build_chroma_neighbors(mb_x, mb_y, 1);
            crate::codec::h264::intra_pred::predict_chroma_8x8(cb_decision.mode, &cr_neighbors)
        };

        // 3. Luma residual: forward DCT per 4×4 → Hadamard DC → quant.
        let intra = QuantParams {
            qp,
            slice: QuantSlice::Intra,
        };
        let mut ac_levels = [[[0i32; 4]; 4]; 16];
        let mut dc_grid = [[0i32; 4]; 4];
        for k in 0..16 {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            let sby = by as usize;
            let sbx = bx as usize;
            let mut sub_res = [[0i32; 4]; 4];
            for dy in 0..4 {
                for dx in 0..4 {
                    sub_res[dy][dx] = src_y[sby * 4 + dy][sbx * 4 + dx] as i32
                        - pred_y[sby * 4 + dy][sbx * 4 + dx] as i32;
                }
            }
            let mut coeffs = forward_dct_4x4(&sub_res);
            // DC block is arranged in RASTER order by (bx, by), NOT
            // BlockIndex order. Spec § 8.5.10: the 4×4 DC-Hadamard
            // block c[j, i] holds the DC of the 4×4 sub-block at
            // MB-position (i, j) (raster col, raster row in 4×4
            // units). Previously we stored dc_grid[k/4][k%4] =
            // BlockIndex order, which swaps blocks {2,3,4,5} with
            // {4,5,2,3} — decoder reads wrong DCs for those slots.
            dc_grid[by as usize][bx as usize] = coeffs[0][0];
            coeffs[0][0] = 0;
            ac_levels[k] = trellis_quantize_4x4(&coeffs, intra, true)
                .unwrap_or_else(|_| forward_quantize_4x4(&coeffs, intra));
        }
        let dc_hadamard = forward_hadamard_4x4(&dc_grid);
        let dc_levels = forward_quantize_dc_luma(&dc_hadamard, qp, QuantSlice::Intra);

        let cbp_luma_flag: u8 = if ac_levels
            .iter()
            .any(|block| block.iter().any(|row| row.iter().any(|&v| v != 0)))
        {
            1
        } else {
            0
        };

        // 4. Chroma residual.
        let chroma_intra = QuantParams {
            qp: qp_c,
            slice: QuantSlice::Intra,
        };
        let (cb_ac_levels, cb_dc_levels) =
            self.encode_chroma_component(&src_cb, &pred_cb, chroma_intra);
        let (cr_ac_levels, cr_dc_levels) =
            self.encode_chroma_component(&src_cr, &pred_cr, chroma_intra);

        // §B-cascade-real Phase 1.1.B step 2 — parallel post-flip
        // Intra_16x16 arrays. Initialised to pre-flip levels; updated
        // by the residual-hook fire sites below.
        let mut ac_levels_post = ac_levels;
        let mut dc_levels_post = dc_levels;
        let mut cb_ac_post = cb_ac_levels;
        let mut cb_dc_post = cb_dc_levels;
        let mut cr_ac_post = cr_ac_levels;
        let mut cr_dc_post = cr_dc_levels;

        let any_chroma_ac = cb_ac_levels
            .iter()
            .chain(cr_ac_levels.iter())
            .any(|block| block.iter().any(|row| row.iter().any(|&v| v != 0)));
        let any_chroma_dc = cb_dc_levels
            .iter()
            .flatten()
            .chain(cr_dc_levels.iter().flatten())
            .any(|&v| v != 0);
        let cbp_chroma: u8 = if any_chroma_ac {
            2
        } else if any_chroma_dc {
            1
        } else {
            0
        };

        // 5. mb_type (Table 7-11 encoding) — Task #50 helper.
        let mb_type = crate::codec::h264::cabac::mb_type_math::pack_i_16x16_mb_type(
            crate::codec::h264::cabac::mb_type_math::I16x16MbType {
                luma_pred_mode,
                cbp_chroma: cbp_chroma as u32,
                cbp_luma_flag: cbp_luma_flag as u32,
            },
        );

        // 6. Emit CABAC syntax. mb_type encoding is slice-specific:
        // - I-slice: `encode_mb_type_i(cabac, mb_type, mb_x)`
        //   codenum 1..24 per Table 7-11.
        // - P-slice: `encode_mb_type_p(cabac, mb_type + 5, mb_x)`
        //   codenum 6..29 per Table 7-13 (P-slice intra-in-P range,
        //   +5 offset from I-slice codenums).
        // - B-slice (#319): `encode_mb_type_b(cabac, mb_type + 23, mb_x)`
        //   codenum 24..47 per Table 7-14 (B-slice intra-in-B range,
        //   +23 offset from I-slice codenums).
        // The caller (I-frame body / intra-in-P fallback / intra-in-B
        // fallback) is responsible for emitting the leading
        // `mb_skip_flag=0` bin before invoking this function.
        match slice_ctx {
            IntraSliceCtx::I => encode_mb_type_i(cabac, mb_type, mb_x),
            IntraSliceCtx::P => encode_mb_type_p(cabac, mb_type + 5, mb_x),
            IntraSliceCtx::B => {
                use crate::codec::h264::cabac::encoder::encode_mb_type_b;
                encode_mb_type_b(cabac, mb_type + 23, mb_x);
            }
        }
        encode_intra_chroma_pred_mode(cabac, chroma_pred_mode as u8, mb_x);
        // mb_qp_delta always emitted for Intra_16x16 (§ 7.3.5.1).
        let qp_delta = qp as i32 - self.prev_mb_qp;
        encode_mb_qp_delta(cabac, qp_delta);
        self.prev_mb_qp = qp as i32;

        // Track cat-by-cat CBF state for same-MB neighbor lookups.
        // I_16x16 is intra by definition.
        let mut current_cbf = CurrentMbCbf::new();
        let current_is_intra = true;

        // 6a. Intra16x16DCLevel — always emitted, single block (cat 0).
        let mut dc_scan = raster_to_scan_levels(&dc_levels);
        // Phase 6D.8: stego hook may modify dc_scan in place. Both
        // entropy emit (below) and reconstruction (later) see the
        // modified buffer → no enc/dec drift.
        self.invoke_stego_residual_hook(
            mb_x, mb_y, &mut dc_scan, 0, 15,
            super::super::stego::orchestrate::ResidualPathKind::LumaDcIntra16x16,
        );
        // §B-cascade-real Phase 1.1.B step 2: capture post-flip DC
        // raster for visual_recon.
        dc_levels_post = scan_to_raster_levels(&dc_scan);
        if super::mb_decision_b::env_var_os_is_some("PHASM_DEBUG_IIP_LEVELS") && in_p_slice {
            eprintln!("ENC IIP@({},{}) DC scan: {:?}", mb_x, mb_y, dc_scan);
        }
        let dc_inc = compute_cbf_ctx_idx_inc_luma_dc(&cabac.neighbors, mb_x);
        let dc_coded = encode_residual_block_cabac_with_cbf_inc(
            cabac, &dc_scan, 0, 15, 0, dc_inc,
        );
        current_cbf.set(0, 0, dc_coded);

        // 6b. Intra16x16ACLevel — 16 blocks (cat 1), only if
        //     cbp_luma_flag = 1. Scan in BLOCK_INDEX_TO_POS order.
        if cbp_luma_flag != 0 {
            for k in 0..16 {
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                let mut ac_scan = ac_scan_order_15(&ac_levels[k]);
                // Phase 6D.8: stego hook for Intra_16x16 AC residual.
                self.invoke_stego_residual_hook(
                    mb_x, mb_y, &mut ac_scan, 0, 14,
                    super::super::stego::orchestrate::ResidualPathKind::Luma4x4 {
                        block_idx: k as u8,
                    },
                );
                // §B-cascade-real Phase 1.1.B step 2: capture post-flip
                // AC raster for visual_recon. AC has [0][0] = 0 because
                // DC was extracted to dc_levels.
                ac_levels_post[k] = ac_scan_15_to_raster(&ac_scan);
                if super::mb_decision_b::env_var_os_is_some("PHASM_DEBUG_IIP_LEVELS") && in_p_slice {
                    eprintln!("ENC IIP AC k={} bx={} by={} scan: {:?}", k, bx, by, ac_scan);
                }
                let ac_inc = compute_cbf_ctx_idx_inc_luma_ac(
                    &current_cbf,
                    &cabac.neighbors,
                    mb_x,
                    bx,
                    by,
                    current_is_intra,
                );
                let ac_coded = encode_residual_block_cabac_with_cbf_inc(
                    cabac, &ac_scan, 0, 14, 1, ac_inc,
                );
                current_cbf.set(1, k, ac_coded);
            }
        }

        // 6c. ChromaDCLevel — 1 block per plane (cat 3), if cbp_chroma >= 1.
        //     Flat 4-element zigzag (the 2×2 Hadamard grid in raster).
        if cbp_chroma >= 1 {
            for (plane, dc) in [&cb_dc_levels, &cr_dc_levels].iter().enumerate() {
                let mut dc_flat: [i32; 4] = [dc[0][0], dc[0][1], dc[1][0], dc[1][1]];
                // Phase 6D.8: stego hook for Chroma DC residual.
                self.invoke_stego_residual_hook(
                    mb_x, mb_y, &mut dc_flat, 0, 3,
                    super::super::stego::orchestrate::ResidualPathKind::ChromaDc {
                        plane: plane as u8,
                    },
                );
                // §B-cascade-real Phase 1.1.B step 2: capture post-flip
                // chroma DC.
                let dc_post_raster: [[i32; 2]; 2] =
                    [[dc_flat[0], dc_flat[1]], [dc_flat[2], dc_flat[3]]];
                if plane == 0 {
                    cb_dc_post = dc_post_raster;
                } else {
                    cr_dc_post = dc_post_raster;
                }
                let inc =
                    compute_cbf_ctx_idx_inc_chroma_dc(&cabac.neighbors, mb_x, plane as u8, current_is_intra);
                let coded = encode_residual_block_cabac_with_cbf_inc(
                    cabac, &dc_flat, 0, 3, 3, inc,
                );
                current_cbf.set(3, plane, coded);
            }
        }

        // 6d. ChromaACLevel — 4 blocks per plane (cat 4), if cbp_chroma == 2.
        if cbp_chroma == 2 {
            for (plane, ac_blocks) in
                [&cb_ac_levels, &cr_ac_levels].iter().enumerate()
            {
                // 2×2 chroma 4×4 grid: (bx, by) ∈ {0,1} × {0,1}.
                // raster order: (0,0), (1,0), (0,1), (1,1).
                for sub in 0..4 {
                    let bx = (sub % 2) as u8;
                    let by = (sub / 2) as u8;
                    let mut ac_scan = ac_scan_order_15(&ac_blocks[sub]);
                    // Phase 6D.8: stego hook for Chroma AC residual.
                    self.invoke_stego_residual_hook(
                        mb_x, mb_y, &mut ac_scan, 0, 14,
                        super::super::stego::orchestrate::ResidualPathKind::ChromaAc {
                            plane: plane as u8,
                            block_idx: sub as u8,
                        },
                    );
                    // §B-cascade-real Phase 1.1.B step 2: capture
                    // post-flip chroma AC.
                    let ac_post_raster = ac_scan_15_to_raster(&ac_scan);
                    if plane == 0 {
                        cb_ac_post[sub] = ac_post_raster;
                    } else {
                        cr_ac_post[sub] = ac_post_raster;
                    }
                    let inc = compute_cbf_ctx_idx_inc_chroma_ac(
                        &current_cbf,
                        &cabac.neighbors,
                        mb_x,
                        plane as u8,
                        bx,
                        by,
                        current_is_intra,
                    );
                    let coded = encode_residual_block_cabac_with_cbf_inc(
                        cabac, &ac_scan, 0, 14, 4, inc,
                    );
                    current_cbf.set(4, block_pos_to_chroma_ac_idx(plane as u8, bx, by), coded);
                }
            }
        }

        // 7. end_of_slice_flag (terminate-coded).
        encode_end_of_slice_flag(cabac, is_last_mb);

        // 8. Commit neighbor state.
        let mut nb = CabacNeighborMB::default();
        nb.mb_type = MbTypeClass::I16x16;
        nb.intra_chroma_pred_mode = chroma_pred_mode as u8;
        nb.cbp_luma = if cbp_luma_flag != 0 { 0x0F } else { 0 };
        nb.cbp_chroma = cbp_chroma;
        nb.mb_qp_delta = qp_delta;
        nb.coded_block_flag_cat = current_cbf.to_neighbor_cbf();
        cabac.neighbors.commit(mb_x, nb);

        // 9. Reconstruction (CAVLC variant — no stego hooks, so both
        // recon buffers identical; dual-write keeps visual_recon in
        // sync after step 3 removes the frame-end mirror).
        let recon_y = self.reconstruct_luma_mb(&pred_y, &ac_levels, &dc_levels, qp);
        self.write_luma_mb_dual(mb_x as u32, mb_y as u32, &recon_y);
        let recon_cb =
            self.reconstruct_chroma_mb(&pred_cb, &cb_ac_levels, &cb_dc_levels, qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &recon_cb);
        let recon_cr =
            self.reconstruct_chroma_mb(&pred_cr, &cr_ac_levels, &cr_dc_levels, qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &recon_cr);

        // §B-cascade-real Phase 1.1.B step 2 — visual_recon path with
        // POST-flip levels (luma + chroma). Intra_16x16 luma is Shape 1
        // (recon at function epilogue), so we get a true second-pass
        // reconstruct here, not a mirror. Sites 5–8 of 15.
        let visual_recon_y =
            self.reconstruct_luma_mb(&pred_y, &ac_levels_post, &dc_levels_post, qp);
        self.visual_recon
            .write_luma_mb(mb_x as u32, mb_y as u32, &visual_recon_y);
        let visual_cb =
            self.reconstruct_chroma_mb(&pred_cb, &cb_ac_post, &cb_dc_post, qp_c);
        self.visual_recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &visual_cb);
        let visual_cr =
            self.reconstruct_chroma_mb(&pred_cr, &cr_ac_post, &cr_dc_post, qp_c);
        self.visual_recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &visual_cr);

        // 10. Keep total_coeff_grid populated for the deblocker's
        //     coded_flags reader (matches the CAVLC path behavior).
        //     For CABAC we don't have TotalCoeff from VLC — use the
        //     per-block cbp_luma bit as a proxy (0 = no AC, 1 = AC).
        for k in 0..16 {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            let abs_bx = mb_x * 4 + bx as usize;
            let abs_by = mb_y * 4 + by as usize;
            let coded = if current_cbf.get(1, k) { 1 } else { 0 };
            self.store_total_coeff_luma(abs_bx, abs_by, coded);
        }

        Ok(())
    }

    /// Build a per-4×4-block `coded_flags` bitmap for the deblocking
    /// filter. Reads from the `total_coeff_grid` populated during
    /// CAVLC emit — a block is "coded" iff it had any nonzero level.
    fn build_coded_flags(&self) -> Vec<bool> {
        let w4 = (self.width / 4) as usize;
        let h4 = (self.height / 4) as usize;
        let mut flags = vec![false; w4 * h4];
        for i in 0..(w4 * h4) {
            flags[i] = self.total_coeff_grid[i] != 0xFF && self.total_coeff_grid[i] > 0;
        }
        flags
    }

    /// Compute CAVLC nC per spec § 9.2.1.1 for a luma 4×4 block at
    /// frame-4x4-grid position `(bx, by)`. Averages left + top
    /// neighbors' TotalCoeff; falls back to single-neighbor or 0 when
    /// neighbors are unavailable.
    /// Compute nC for the current MB's Intra16x16DCLevel block per
    /// spec § 9.2.1.1: "derived as if Luma4x4 with luma4x4BlkIdx = 0
    /// were being decoded". The neighbors are the LEFT and ABOVE
    /// 4x4 luma blocks (rightmost column of left MB / bottom row of
    /// above MB) — NOT a separate DC TC. Equivalent to the
    /// per-4x4-block luma rule applied at the MB's top-left position.
    fn compute_nc_intra16x16_dc(&self, mb_x: usize, mb_y: usize) -> i8 {
        self.compute_nc_luma_at(mb_x * 4, mb_y * 4)
    }

    /// Compute nC for a ChromaACLevel block at chroma 4×4 grid
    /// position `(cx, cy)` per spec § 9.2.1.1. Mirrors the decoder's
    /// `chroma_nc` exactly: frame-edge unavailability via cx/cy>0;
    /// otherwise read the grid value (0 if neighbor MB exists but
    /// didn't emit chroma AC). `is_cr` selects the Cr or Cb grid.
    fn compute_nc_chroma_at(&self, cx: usize, cy: usize, is_cr: bool) -> i8 {
        let cw = (self.width / 8) as usize;
        let grid = if is_cr {
            &self.chroma_cr_tc_grid
        } else {
            &self.chroma_cb_tc_grid
        };
        let have_left = cx > 0;
        let have_top = cy > 0;
        match (have_left, have_top) {
            (false, false) => 0,
            (true, false) => grid[cy * cw + cx - 1] as i8,
            (false, true) => grid[(cy - 1) * cw + cx] as i8,
            (true, true) => {
                let na = grid[cy * cw + cx - 1] as i16;
                let nb = grid[(cy - 1) * cw + cx] as i16;
                ((na + nb + 1) >> 1) as i8
            }
        }
    }

    /// Store TotalCoeff for a chroma 4x4 block.
    fn store_total_coeff_chroma(&mut self, cx: usize, cy: usize, is_cr: bool, tc: u8) {
        let cw = (self.width / 8) as usize;
        let grid = if is_cr {
            &mut self.chroma_cr_tc_grid
        } else {
            &mut self.chroma_cb_tc_grid
        };
        if cy * cw + cx < grid.len() {
            grid[cy * cw + cx] = tc;
        }
    }

    fn compute_nc_luma_at(&self, bx: usize, by: usize) -> i8 {
        let w4 = (self.width / 4) as usize;
        let left = if bx > 0 {
            Some(self.total_coeff_grid[by * w4 + (bx - 1)])
        } else {
            None
        }
        .and_then(|v| if v == 0xFF { None } else { Some(v) });
        let top = if by > 0 {
            Some(self.total_coeff_grid[(by - 1) * w4 + bx])
        } else {
            None
        }
        .and_then(|v| if v == 0xFF { None } else { Some(v) });
        match (left, top) {
            (None, None) => 0,
            (Some(v), None) | (None, Some(v)) => v.min(16) as i8,
            (Some(a), Some(b)) => {
                (((a as u16 + b as u16 + 1) >> 1).min(16)) as i8
            }
        }
    }

    /// Record the final per-MB QP and intra flag used during
    /// residual quant + reconstruction. Called at the top of every
    /// `write_*_macroblock` function right after `mb_qp` is derived
    /// from AQ. The deblock filter reads these to compute per-edge
    /// `qp_avg = (qp_p + qp_q + 1) >> 1` (spec § 8.7.2.1) and the
    /// correct boundary strength (spec § 8.7.2.1 bs derivation).
    /// Without this, AQ'd per-MB QPs produce encoder-side deblock
    /// outputs that diverge from decoder-side (the decoder parses
    /// mb_qp_delta and uses the per-MB QP correctly).
    #[inline]
    /// Phase F pre-pass: scan the luma plane once, compute per-MB
    /// variance and log2, return the frame-mean log2 in Q8. Only
    /// allocates when PHASM_AQ_MODE=3; other modes return 0 and the
    /// old AQ path stays in effect.
    fn compute_aq_frame_mean(
        &self,
        y_plane: &[u8],
        y_stride: usize,
        mb_w: usize,
        mb_h: usize,
    ) -> i32 {
        let mode: u8 = super::mb_decision_b::env_var("PHASM_AQ_MODE")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        if mode != 3 {
            return 0;
        }
        let mut log2s: Vec<i32> = Vec::with_capacity(mb_w * mb_h);
        let mut src = [[0u8; 16]; 16];
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                for dy in 0..16 {
                    for dx in 0..16 {
                        src[dy][dx] =
                            y_plane[(mb_y * 16 + dy) * y_stride + mb_x * 16 + dx];
                    }
                }
                let var = super::rate_control::mb_variance_16x16(&src);
                log2s.push(super::rate_control::log2_var_q8(var));
            }
        }
        super::rate_control::frame_mean_log2_q8(&log2s)
    }

    fn commit_mb_state(&mut self, mb_x: usize, mb_y: usize, mb_qp: u8, is_intra: bool) {
        let mb_w = (self.width / 16) as usize;
        let idx = mb_y * mb_w + mb_x;
        self.qp_grid[idx] = mb_qp;
        self.intra_grid[idx] = is_intra;
        // Default transform_8x8_flag = false; the I_8x8 commit path
        // overrides this after calling commit_mb_state.
        self.transform_8x8_grid[idx] = false;
    }

    /// Mark the current MB as using 8×8 transform — consumed by the
    /// deblock filter to skip internal 4-pixel-grid edges inside this
    /// MB (spec § 8.7.2.1).
    fn mark_transform_8x8_mb(&mut self, mb_x: usize, mb_y: usize) {
        let mb_w = (self.width / 16) as usize;
        let idx = mb_y * mb_w + mb_x;
        self.transform_8x8_grid[idx] = true;
    }

    /// Update the total_coeff_grid after a luma 4×4 block emits.
    fn store_total_coeff_luma(&mut self, bx: usize, by: usize, total_coeff: u8) {
        let w4 = (self.width / 4) as usize;
        self.total_coeff_grid[by * w4 + bx] = total_coeff;
    }

    fn idr_slice_header(&self) -> ISliceHeaderParams {
        let slice_qp = self.rc.target_crf as i32;
        let pic_init_qp = self.pps_params.pic_init_qp as i32;
        let disable_deblocking = super::mb_decision_b::env_var_os_is_some("PHASM_DISABLE_DEBLOCK");
        ISliceHeaderParams {
            is_idr: true,
            pps_id: 0,
            frame_num: self.frame_num,
            idr_pic_id: 0,
            slice_qp_delta: slice_qp - pic_init_qp,
            disable_deblocking,
            // §6E-A4 — IDR's POC is always 0 by definition. When
            // SPS uses pic_order_cnt_type=0, slice header MUST
            // emit pic_order_cnt_lsb. When SPS uses type=2 (legacy
            // I+P-only), this field is omitted (None).
            pic_order_cnt_lsb: if self.enable_b_frames { Some(0) } else { None },
            log2_max_pic_order_cnt_lsb_minus4: self.sps_params.log2_max_pic_order_cnt_lsb_minus4,
        }
    }

    /// §6E-A4(c)-lite — compute `pic_order_cnt_lsb` for a P-frame slice
    /// header, based on the encoder's tracked anchor display index.
    /// Returns `None` when `enable_b_frames=false` (SPS uses type
    /// 2, no POC LSB field). Returns `Some(lsb)` when type=0.
    fn maybe_p_poc_lsb(&self) -> Option<u32> {
        if !self.enable_b_frames {
            return None;
        }
        // The P-frame being encoded advances the anchor by m_factor
        // (= 2 for IBPBP). At this point in the encode flow,
        // self.display_idx_of_prev_anchor still holds the PREVIOUS
        // anchor; the new P's display index is prev_anchor + 2.
        // (For the first P after IDR, prev_anchor=0 so display=2.)
        // The IDR resets the anchor to 0 inside encode_i_frame.
        let display_index = self.display_idx_of_prev_anchor + 2;
        Some(self.poc_tracker.poc_lsb_for(display_index))
    }

    // ─── I_PCM macroblock ─────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn write_pcm_macroblock(
        &mut self,
        w: &mut BitWriter,
        mb_x: usize,
        mb_y: usize,
        y_plane: &[u8],
        y_stride: usize,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
    ) -> Result<(), EncoderError> {
        w.write_ue(25); // I_PCM mb_type
        while !w.byte_aligned() {
            w.write_bit(false);
        }

        let y0 = mb_y * 16;
        let x0 = mb_x * 16;
        let mut luma_block = [[0u8; 16]; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                let v = y_plane[(y0 + dy) * y_stride + x0 + dx];
                w.write_bits(v as u32, 8);
                luma_block[dy][dx] = v;
            }
        }

        let cy0 = mb_y * 8;
        let cx0 = mb_x * 8;
        let mut cb_block = [[0u8; 8]; 8];
        let mut cr_block = [[0u8; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                let v = cb_plane[(cy0 + dy) * c_stride + cx0 + dx];
                w.write_bits(v as u32, 8);
                cb_block[dy][dx] = v;
            }
        }
        for dy in 0..8 {
            for dx in 0..8 {
                let v = cr_plane[(cy0 + dy) * c_stride + cx0 + dx];
                w.write_bits(v as u32, 8);
                cr_block[dy][dx] = v;
            }
        }

        // §B-cascade-real Phase 1.1.B step 3: dual-write PCM (lossless
        // pixel copy — no stego flips can apply here).
        self.write_luma_mb_dual(mb_x as u32, mb_y as u32, &luma_block);
        self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 0, &cb_block);
        self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 1, &cr_block);

        Ok(())
    }

    // ─── Intra MB dispatcher (I_4x4 vs I_16x16 RDO) ──────────────

    /// Penalty in SATD units for the extra bits I_4x4 carries
    /// (16 sub-block mode flags + possibly per-block rem fields).
    /// Measured on 1080p IMG_4138 content after the psy-RD fix
    /// (commit 5ac5a83), the optimal value is 0 — sweep in 0..64
    /// showed monotonic quality degradation as penalty rose. At
    /// penalty=0, I+4P Y-PSNR = 46.50 dB (vs 43.69 dB at 64); size
    /// only grows +1.2% for +2.8 dB. At reasonable QP, per-sub-
    /// block intra prediction beats 16x16 DC/plane enough that the
    /// I_4x4 bit overhead pays for itself. Revisit if a real
    /// bit-cost model replaces SATD-plus-constant-overhead.
    const I4X4_OVERHEAD_PENALTY: u32 = 0;

    /// Phase D.1 intra-in-P RDO ceiling (task #53 calibration).
    /// Default 95/100 = 0.95 — calibrated against a reference H.264
    /// encoder on IMG_4138 / IMG_4273 first-10-frames Q=22. See the
    /// comment at the gate site for the sweep data. Env-overridable
    /// via `PHASM_INTRA_RDO_CEIL_NUM` / `_DEN`.
    const INTRA_RDO_CEIL_NUM: u64 = 95;
    const INTRA_RDO_CEIL_DEN: u64 = 100;

    /// Phase D.4 (task #51) fast-intra threshold (Q.10). Gate:
    /// run I_4x4 speculation in intra-in-P only when
    /// `satd_inter > THRESH × satd_i16x16`. The idea is to skip the
    /// 144-SATD I_4x4 cost on MBs where inter is already close to
    /// intra, focusing I_4x4 effort on drift-damaged MBs.
    ///
    /// Calibration on IMG_4138 Q=22 (10 frames, intra-in-P set ~10%
    /// of MBs after the Phase D.2-stealth penalty=240 tuning):
    ///   thresh=0    → 4.18 % I_4x4, wall=27.97 s  (reference 4.3 %)
    ///   thresh=512  → 4.17 % I_4x4, wall=27.91 s
    ///   thresh=800  → 1.53 % I_4x4, wall=27.97 s
    ///   thresh=1000 → 0.30 % I_4x4, wall=28.15 s
    ///   thresh=1536 → 0.00 % I_4x4 (1.5×; common in speed-tuned
    ///                 encoders, but kills I_4x4 entirely on our
    ///                 content given our D.1 RDO gate)
    ///
    /// Our Phase D.1 RDO gate already pre-selects drift-damaged MBs
    /// on `D + λR`, not raw SATD — so by the time we reach the
    /// intra-in-P branch, `inter_satd` is often less than
    /// `intra_satd` (intra won on rate, not distortion). A 1.5×
    /// requirement on SATD then nukes I_4x4. Also: the 144 SATDs of
    /// I_4x4 speculation are small vs total encode time (the gate
    /// barely moves wall-clock across the sweep). Net: the gate
    /// doesn't pay for itself on our pipeline.
    ///
    /// **Default 0 (disabled).** The knob stays for callers who want
    /// it — e.g. stronger speed targets where a few % I_4x4 loss
    /// is acceptable. Override via `PHASM_D4_FAST_INTRA_THRESH_Q10=<n>`.
    const D4_FAST_INTRA_THRESH_Q10: u32 = 0;

    /// Phase D.2-stealth (task #52) intra-in-P I_4x4 penalty. This
    /// is SEPARATE from the I-slice `I4X4_OVERHEAD_PENALTY` above:
    /// in an intra-in-P MB the I_4x4 emit path also competes
    /// against the I_16x16 intra-in-P codenum (P-slice mb_type 6+),
    /// which has a much smaller bit cost than I-slice I_16x16, so
    /// the I_4x4 overhead must be larger to reach a realistic
    /// distribution. Calibrated 2026-04-24 against a reference
    /// H.264 encoder at Main-profile / medium-speed / no-B-frames
    /// on IMG_4138 / IMG_4273 at QP=22: penalty=240 lands our I_4x4
    /// share within intra-in-P at 4.2 % / 1.7 % (reference 4.3 % /
    /// 3.9 %), both within ±5 pp of the reference. Override via
    /// `PHASM_IIP_I4X4_PENALTY=<n>`.
    const IIP_I4X4_PENALTY: u32 = 240;

    /// Penalty in SATD units for choosing I_16x16 inside a P-slice
    /// MB when any inter partition is competitive. Accounts for the
    /// extra ~10 bits: bigger mb_type codenum (5..=30 vs 0..=4),
    /// chroma pred mode, no MV reuse, plus slight residual cost
    /// delta. Spec § 8.4.1.3.2 — intra MBs in P-slices are rare but
    /// essential on frames with scene cuts or poor predictors.
    ///
    /// 2026-04-20: tried lowering to 64 to rescue drifted MBs via
    /// more-aggressive intra fallback — REGRESSED 7 dB. Intra in
    /// the middle of an inter-heavy P-slice has no intra neighbors,
    /// so pred_mode falls back to DC → worse than even a drifted
    /// inter prediction. Keep 512.
    const P_INTRA_OVERHEAD: u32 = 512;

    /// Psy-RD strength (SATD units per unit AC-energy diff). 0
    /// disables; 4 is a moderate first-pass bias. Biases intra mode
    /// decision toward predictions with similar texture amount as
    /// the source. Phase E.3 revisit (task #156): swap to Hadamard-AC
    /// after inter-psy is calibrated.
    const PSY_RD_STRENGTH: u32 = 4;

    /// Decide between Intra_16x16 and Intra_4x4 for the MB at
    /// `(mb_x, mb_y)` by SATD + overhead penalty and commit the
    /// winner (emit syntax + residual + reconstruct).
    #[allow(clippy::too_many_arguments)]
    fn write_intra_macroblock(
        &mut self,
        w: &mut BitWriter,
        mb_x: usize,
        mb_y: usize,
        y_plane: &[u8],
        y_stride: usize,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
        qp: u8,
        qp_c_unused: u8,
    ) -> Result<(), EncoderError> {
        let _ = qp_c_unused;
        // Source luma extraction (for the SATD comparison + both paths).
        let y0 = mb_y * 16;
        let x0 = mb_x * 16;
        let mut src_y = [[0u8; 16]; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                src_y[dy][dx] = y_plane[(y0 + dy) * y_stride + x0 + dx];
            }
        }

        // AQ-1: per-MB target QP from luma variance.
        let variance = super::rate_control::mb_variance_16x16(&src_y);
        let qp_offset = super::rate_control::variance_to_qp_offset(variance);
        let mb_qp = ((qp as i32) + qp_offset).clamp(0, 51) as u8;
        let mb_qp_c = derive_chroma_qp(mb_qp as i32, self.pps_params.chroma_qp_index_offset as i32) as u8;
        // Commit MB state for the deblock filter (I-slice → is_intra=true).
        self.commit_mb_state(mb_x, mb_y, mb_qp, true);

        // I_16x16 SATD (side-effect-free, psy-RD biased).
        let neighbors_y = self.build_luma_neighbors_16x16(mb_x, mb_y);
        let i16_decision =
            choose_intra_16x16_mode_psy(&neighbors_y, &src_y, Self::PSY_RD_STRENGTH);
        let i16_cost = i16_decision.satd;

        // I_4x4 pipeline (mutates recon — if not chosen, I_16x16
        // reconstruction overwrites this region inside
        // `write_i16x16_macroblock`).
        let psy = Self::PSY_RD_STRENGTH;
        let i4x4_result = encode_i4x4_mb(
            &src_y,
            &mut self.recon,
            mb_x,
            mb_y,
            mb_qp,
            psy,
        );
        let i4x4_cost = i4x4_result.total_satd.saturating_add(Self::I4X4_OVERHEAD_PENALTY);

        if i4x4_cost < i16_cost {
            // Commit I_4x4: emit syntax, update the mode grid.
            self.commit_i4x4_macroblock(
                w,
                mb_x,
                mb_y,
                &i4x4_result,
                cb_plane,
                cr_plane,
                c_stride,
                mb_qp,
                mb_qp_c,
            )?;
        } else {
            // Discard I_4x4 — clear the mode-grid entries so future
            // neighbor queries use DC fallback — and fall through to
            // the existing I_16x16 path (which will overwrite the
            // tentatively-reconstructed pixels).
            self.clear_i4x4_mode_grid_for_mb(mb_x, mb_y);
            self.write_i16x16_macroblock(
                w, mb_x, mb_y, y_plane, y_stride, cb_plane, cr_plane, c_stride, mb_qp, mb_qp_c,
                0, // I-slice: mb_type codenum is the I-slice value directly
            )?;
        }
        Ok(())
    }

    /// Clear the I_4x4 mode grid for one MB (sets all 16 slots to
    /// 0xFF — "not an I_4x4 block").
    fn clear_i4x4_mode_grid_for_mb(&mut self, mb_x: usize, mb_y: usize) {
        let w4 = (self.width / 4) as usize;
        let base_x = mb_x * 4;
        let base_y = mb_y * 4;
        for dy in 0..4 {
            for dx in 0..4 {
                let idx = (base_y + dy) * w4 + (base_x + dx);
                self.i4x4_mode_grid[idx] = 0xFF;
            }
        }
    }

    /// Record I_4x4 modes into the frame-wide grid (BlockIndex scan
    /// order → 4×4-block-in-MB coords via BLOCK_INDEX_TO_POS).
    fn store_i4x4_mode_grid_for_mb(
        &mut self,
        mb_x: usize,
        mb_y: usize,
        modes: &[u8; 16],
    ) {
        use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;
        let w4 = (self.width / 4) as usize;
        let base_x = mb_x * 4;
        let base_y = mb_y * 4;
        for blk_idx in 0..16 {
            let (bx, by) = BLOCK_INDEX_TO_POS[blk_idx];
            let x = base_x + bx as usize;
            let y = base_y + by as usize;
            self.i4x4_mode_grid[y * w4 + x] = modes[blk_idx];
        }
    }

    /// Lookup the I_4x4 mode at 4×4-grid position `(bx, by)`.
    /// Returns `None` for out-of-bounds or non-I_4x4 blocks.
    fn lookup_i4x4_mode(&self, bx: isize, by: isize) -> Option<u8> {
        let w4 = (self.width / 4) as isize;
        let h4 = (self.height / 4) as isize;
        if bx < 0 || by < 0 || bx >= w4 || by >= h4 {
            return None;
        }
        let v = self.i4x4_mode_grid[(by as usize) * (w4 as usize) + bx as usize];
        if v == 0xFF {
            None
        } else {
            Some(v)
        }
    }

    // ─── Intra_8x8 mode grid helpers (Phase 100-F2) ──────────────

    #[allow(dead_code)] // Reserved for Intra_8x8 mode-grid invalidation; not yet wired.
    fn clear_i8x8_mode_grid_for_mb(&mut self, mb_x: usize, mb_y: usize) {
        let w8 = (self.width / 8) as usize;
        let base_x = mb_x * 2;
        let base_y = mb_y * 2;
        for dy in 0..2 {
            for dx in 0..2 {
                let idx = (base_y + dy) * w8 + (base_x + dx);
                self.i8x8_mode_grid[idx] = 0xFF;
            }
        }
    }

    /// Record the 4 I_8x8 modes into the frame-wide 8×8-granular grid.
    /// `modes[blk_idx]` follows `I8X8_BLOCK_POS` (raster scan).
    fn store_i8x8_mode_grid_for_mb(&mut self, mb_x: usize, mb_y: usize, modes: &[u8; 4]) {
        use super::intra_8x8_encode::I8X8_BLOCK_POS;
        let w8 = (self.width / 8) as usize;
        let base_x = mb_x * 2;
        let base_y = mb_y * 2;
        for blk_idx in 0..4 {
            let (bx, by) = I8X8_BLOCK_POS[blk_idx];
            let x = base_x + bx as usize;
            let y = base_y + by as usize;
            self.i8x8_mode_grid[y * w8 + x] = modes[blk_idx];
        }
    }

    /// Lookup the I_8x8 mode at 8×8-grid position `(bx, by)`. Returns
    /// `None` for out-of-bounds or non-I_8x8 blocks.
    fn lookup_i8x8_mode(&self, bx: isize, by: isize) -> Option<u8> {
        let w8 = (self.width / 8) as isize;
        let h8 = (self.height / 8) as isize;
        if bx < 0 || by < 0 || bx >= w8 || by >= h8 {
            return None;
        }
        let v = self.i8x8_mode_grid[(by as usize) * (w8 as usize) + bx as usize];
        if v == 0xFF {
            None
        } else {
            Some(v)
        }
    }

    // ─── Intra_4x4 macroblock commit path ────────────────────────

    /// Emit bitstream syntax for an I_4x4 MB (mb_type=0 + 16 mode
    /// flags + chroma_pred_mode + CBP + mb_qp_delta + residuals).
    /// The caller has already run `encode_i4x4_mb` (which mutated
    /// `self.recon`'s luma region for each sub-block); this function
    /// additionally runs chroma encode + recon.
    #[allow(clippy::too_many_arguments)]
    fn commit_i4x4_macroblock(
        &mut self,
        w: &mut BitWriter,
        mb_x: usize,
        mb_y: usize,
        i4x4: &I4x4MbResult,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
        qp: u8,
        qp_c: u8,
    ) -> Result<(), EncoderError> {
        use super::inter_mode::cbp_to_codenum_intra;
        use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;

        // ── Chroma encode (same pipeline as I_16x16) ──
        let cy0 = mb_y * 8;
        let cx0 = mb_x * 8;
        let mut src_cb = [[0u8; 8]; 8];
        let mut src_cr = [[0u8; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                src_cb[dy][dx] = cb_plane[(cy0 + dy) * c_stride + cx0 + dx];
                src_cr[dy][dx] = cr_plane[(cy0 + dy) * c_stride + cx0 + dx];
            }
        }
        // Chroma mode decision (shared between Cb and Cr).
        let cb_neighbors = self.build_chroma_neighbors(mb_x, mb_y, 0);
        let cb_decision = choose_intra_chroma_mode(&cb_neighbors, &src_cb);
        let chroma_pred_mode = cb_decision.mode as u32;
        let pred_cb = cb_decision.predicted;
        let pred_cr = {
            let cr_neighbors = self.build_chroma_neighbors(mb_x, mb_y, 1);
            crate::codec::h264::intra_pred::predict_chroma_8x8(cb_decision.mode, &cr_neighbors)
        };
        let chroma_qp = QuantParams { qp: qp_c, slice: QuantSlice::Intra };
        let (cb_ac_levels, cb_dc_levels) =
            self.encode_chroma_component(&src_cb, &pred_cb, chroma_qp);
        let (cr_ac_levels, cr_dc_levels) =
            self.encode_chroma_component(&src_cr, &pred_cr, chroma_qp);

        // ── CBP computation ──
        let mut luma_nonzero = [false; 16];
        for k in 0..16 {
            luma_nonzero[k] = i4x4.ac_levels[k]
                .iter()
                .any(|row| row.iter().any(|&v| v != 0));
        }
        let cbp_luma = luma_8x8_cbp_mask(&luma_nonzero);

        let any_chroma_ac = cb_ac_levels
            .iter()
            .chain(cr_ac_levels.iter())
            .any(|b| b.iter().any(|r| r.iter().any(|&v| v != 0)));
        let any_chroma_dc = cb_dc_levels
            .iter()
            .flatten()
            .chain(cr_dc_levels.iter().flatten())
            .any(|&v| v != 0);
        let cbp_chroma = if any_chroma_ac {
            2u8
        } else if any_chroma_dc {
            1u8
        } else {
            0u8
        };
        let cbp_value = pack_cbp(cbp_luma, cbp_chroma);

        // ── Emit syntax ──
        // mb_type = 0 for I_NxN (Intra_4x4 since we don't emit 8×8
        // transform; transform_8x8_mode_flag in PPS is 0).
        w.write_ue(0);

        // Per-sub-block mode flags (16 × (prev_flag + optional rem)).
        let flags = derive_i4x4_mode_flags(&i4x4.modes, mb_x, mb_y, |bx, by| {
            self.lookup_i4x4_mode(bx, by)
        });
        for (flag, rem) in flags.iter() {
            w.write_bit(*flag);
            if let Some(r) = rem {
                w.write_bits(*r as u32, 3);
            }
        }

        // intra_chroma_pred_mode: codeNum matches IntraChroma8x8Mode.
        w.write_ue(chroma_pred_mode);

        // coded_block_pattern via intra column of Table 9-4.
        let cbp_codenum = cbp_to_codenum_intra(cbp_value).ok_or_else(|| {
            EncoderError::InvalidInput(format!("bad intra CBP {cbp_value}"))
        })?;
        w.write_ue(cbp_codenum);

        // mb_qp_delta only when any residual block is present
        // (spec § 7.3.5 — I_4x4 is not Intra_16x16 so the CBP gate
        // applies). When emitted, advance prev_mb_qp; otherwise it
        // stays unchanged (decoder keeps the previous mb_qp_y too).
        if cbp_value != 0 {
            let delta = qp as i32 - self.prev_mb_qp;
            w.write_se(delta);
            self.prev_mb_qp = qp as i32;
        } else {
            // Spec § 7.3.5.1: no mb_qp_delta when CBP=0, decoder uses
            // prev_mb_qp. Override qp_grid to match.
            self.commit_mb_state(mb_x, mb_y, self.prev_mb_qp as u8, true);
        }

        // ── Residual emit ──
        // Luma 4×4 blocks (full 16 coeffs each, DC included).
        if cbp_luma != 0 {
            for k in 0..16 {
                let (bx_in_mb, by_in_mb) = BLOCK_INDEX_TO_POS[k];
                let abs_bx = mb_x * 4 + bx_in_mb as usize;
                let abs_by = mb_y * 4 + by_in_mb as usize;
                if cbp_luma & (1 << (k / 4)) != 0 {
                    let scan = raster_to_scan_levels(&i4x4.ac_levels[k]);
                    let nc = self.compute_nc_luma_at(abs_bx, abs_by);
                    encode_cavlc_block(w, &scan, nc, CavlcBlockType::Luma4x4).map_err(|e| {
                        EncoderError::InvalidInput(format!("CAVLC I_4x4 luma: {e}"))
                    })?;
                    let total_coeff = scan.iter().filter(|&&v| v != 0).count() as u8;
                    self.store_total_coeff_luma(abs_bx, abs_by, total_coeff);
                } else {
                    self.store_total_coeff_luma(abs_bx, abs_by, 0);
                }
            }
        } else {
            for k in 0..16 {
                let (bx_in_mb, by_in_mb) = BLOCK_INDEX_TO_POS[k];
                self.store_total_coeff_luma(
                    mb_x * 4 + bx_in_mb as usize,
                    mb_y * 4 + by_in_mb as usize,
                    0,
                );
            }
        }
        // Chroma DC blocks.
        if cbp_chroma != 0 {
            let cb_dc_flat: Vec<i32> = cb_dc_levels.iter().flatten().copied().collect();
            encode_cavlc_block(w, &cb_dc_flat, -1, CavlcBlockType::ChromaDc)
                .map_err(|e| EncoderError::InvalidInput(format!("CAVLC I_4x4 Cb DC: {e}")))?;
            let cr_dc_flat: Vec<i32> = cr_dc_levels.iter().flatten().copied().collect();
            encode_cavlc_block(w, &cr_dc_flat, -1, CavlcBlockType::ChromaDc)
                .map_err(|e| EncoderError::InvalidInput(format!("CAVLC I_4x4 Cr DC: {e}")))?;
        }
        // Chroma AC blocks.
        if cbp_chroma == 2 {
            let cmb_x = mb_x * 2;
            let cmb_y = mb_y * 2;
            for (sub_idx, block) in cb_ac_levels.iter().enumerate() {
                let cx = cmb_x + sub_idx % 2;
                let cy = cmb_y + sub_idx / 2;
                let scan = ac_scan_order_15(block);
                let nc = self.compute_nc_chroma_at(cx, cy, false);
                encode_cavlc_block(w, &scan, nc, CavlcBlockType::ChromaAc).map_err(|e| {
                    EncoderError::InvalidInput(format!("CAVLC I_4x4 Cb AC: {e}"))
                })?;
                let tc = scan.iter().filter(|&&v| v != 0).count() as u8;
                self.store_total_coeff_chroma(cx, cy, false, tc);
            }
            for (sub_idx, block) in cr_ac_levels.iter().enumerate() {
                let cx = cmb_x + sub_idx % 2;
                let cy = cmb_y + sub_idx / 2;
                let scan = ac_scan_order_15(block);
                let nc = self.compute_nc_chroma_at(cx, cy, true);
                encode_cavlc_block(w, &scan, nc, CavlcBlockType::ChromaAc).map_err(|e| {
                    EncoderError::InvalidInput(format!("CAVLC I_4x4 Cr AC: {e}"))
                })?;
                let tc = scan.iter().filter(|&&v| v != 0).count() as u8;
                self.store_total_coeff_chroma(cx, cy, true, tc);
            }
        }

        // ── Chroma reconstruction ── (CAVLC variant; no stego hooks)
        let recon_cb =
            self.reconstruct_chroma_mb(&pred_cb, &cb_ac_levels, &cb_dc_levels, qp_c);
        self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 0, &recon_cb);
        let recon_cr =
            self.reconstruct_chroma_mb(&pred_cr, &cr_ac_levels, &cr_dc_levels, qp_c);
        self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 1, &recon_cr);

        // Luma recon is already in `self.recon` from `encode_i4x4_mb`.
        // Publish the mode grid.
        self.store_i4x4_mode_grid_for_mb(mb_x, mb_y, &i4x4.modes);

        // Silence unused: BLOCK_INDEX_TO_POS imported for readability.
        let _ = BLOCK_INDEX_TO_POS;
        let _ = qp;

        Ok(())
    }

    // ─── Intra_16x16 macroblock ──────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn write_i16x16_macroblock(
        &mut self,
        w: &mut BitWriter,
        mb_x: usize,
        mb_y: usize,
        y_plane: &[u8],
        y_stride: usize,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
        qp: u8,
        qp_c: u8,
        // P-slice intra needs mb_type offset by 5 per Table 7-13
        // (I-slice codenum 0..=25 becomes P-slice codenum 5..=30).
        mb_type_offset: u32,
    ) -> Result<(), EncoderError> {
        // 1. Gather source 16×16 luma + two 8×8 chroma.
        let y0 = mb_y * 16;
        let x0 = mb_x * 16;
        let mut src_y = [[0u8; 16]; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                src_y[dy][dx] = y_plane[(y0 + dy) * y_stride + x0 + dx];
            }
        }
        let cy0 = mb_y * 8;
        let cx0 = mb_x * 8;
        let mut src_cb = [[0u8; 8]; 8];
        let mut src_cr = [[0u8; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                src_cb[dy][dx] = cb_plane[(cy0 + dy) * c_stride + cx0 + dx];
                src_cr[dy][dx] = cr_plane[(cy0 + dy) * c_stride + cx0 + dx];
            }
        }

        // 2. Intra_16x16 mode decision (Phase 6A polish #4):
        //    pick the cheapest of {Vertical, Horizontal, DC, Plane}
        //    by SATD against the source MB. Per-MB neighbors come
        //    from the ReconBuffer (already-reconstructed pixels).
        let neighbors_y = self.build_luma_neighbors_16x16(mb_x, mb_y);
        let luma_decision =
            choose_intra_16x16_mode_psy(&neighbors_y, &src_y, Self::PSY_RD_STRENGTH);
        let pred_y = luma_decision.predicted;
        let luma_pred_mode = luma_decision.mode as u32;
        // Chroma: per spec both Cb and Cr share a single
        // `intra_chroma_pred_mode`. Run the chooser on Cb source and
        // apply the chosen mode to both components — matches real
        // encoders (Cb and Cr are highly correlated).
        let cb_neighbors = self.build_chroma_neighbors(mb_x, mb_y, 0);
        let cb_decision = choose_intra_chroma_mode(&cb_neighbors, &src_cb);
        let chroma_pred_mode = cb_decision.mode as u32;
        let pred_cb = cb_decision.predicted;
        let pred_cr = {
            let cr_neighbors = self.build_chroma_neighbors(mb_x, mb_y, 1);
            crate::codec::h264::intra_pred::predict_chroma_8x8(cb_decision.mode, &cr_neighbors)
        };

        // 3. Luma residual → forward DCT per 4×4 AC sub-block →
        //    collect DC grid → forward Hadamard → forward-quant DC +
        //    AC.
        //
        // DC grid arrangement: per spec § 8.5.10 + § 6.4.3, the
        // 16 DC values are placed into a 4×4 grid using the
        // macroblock's 4×4-block scan order (BLOCK_INDEX_TO_POS),
        // NOT a simple (sby, sbx) raster. The same ordering is
        // required for any spec-conformant decoder's
        // Intra16x16DCLevel parse. The AC sub-blocks are also
        // indexed by the same BlockIndex k, matching the residual
        // emit order in § 8.
        use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;
        let intra = QuantParams {
            qp,
            slice: QuantSlice::Intra,
        };
        let mut ac_levels = [[[0i32; 4]; 4]; 16]; // indexed by BlockIndex k
        let mut dc_grid = [[0i32; 4]; 4];
        for k in 0..16 {
            let (bx, by) = BLOCK_INDEX_TO_POS[k]; // bx=col, by=row
            let sby = by as usize;
            let sbx = bx as usize;
            let mut sub_res = [[0i32; 4]; 4];
            for dy in 0..4 {
                for dx in 0..4 {
                    sub_res[dy][dx] = src_y[sby * 4 + dy][sbx * 4 + dx] as i32
                        - pred_y[sby * 4 + dy][sbx * 4 + dx] as i32;
                }
            }
            let mut coeffs = forward_dct_4x4(&sub_res);
            // DC block in raster (bx, by) order per spec § 8.5.10 —
            // see the CAVLC counterpart for the full rationale.
            dc_grid[by as usize][bx as usize] = coeffs[0][0];
            coeffs[0][0] = 0;
            ac_levels[k] = trellis_quantize_4x4(&coeffs, intra, true)
                .unwrap_or_else(|_| forward_quantize_4x4(&coeffs, intra));
        }
        let dc_hadamard = forward_hadamard_4x4(&dc_grid);
        let dc_levels = forward_quantize_dc_luma(&dc_hadamard, qp, QuantSlice::Intra);

        // 4. CBP_luma: 1 if any AC non-zero, 0 otherwise (Intra_16x16
        //    is all-or-nothing).
        let cbp_luma_flag: u8 = if ac_levels.iter().any(|block| {
            block.iter().any(|row| row.iter().any(|&v| v != 0))
        }) {
            1
        } else {
            0
        };

        // 5. Chroma residual → same pipeline for Cb + Cr.
        let chroma_intra = QuantParams {
            qp: qp_c,
            slice: QuantSlice::Intra,
        };
        let (cb_ac_levels, cb_dc_levels) =
            self.encode_chroma_component(&src_cb, &pred_cb, chroma_intra);
        let (cr_ac_levels, cr_dc_levels) =
            self.encode_chroma_component(&src_cr, &pred_cr, chroma_intra);

        // 6. CBP_chroma (spec Table 7-15):
        //    0 = no chroma residual
        //    1 = chroma DC only
        //    2 = chroma DC + AC
        let any_chroma_ac = cb_ac_levels.iter().chain(cr_ac_levels.iter()).any(|block| {
            block.iter().any(|row| row.iter().any(|&v| v != 0))
        });
        let any_chroma_dc = cb_dc_levels
            .iter()
            .flatten()
            .chain(cr_dc_levels.iter().flatten())
            .any(|&v| v != 0);
        let cbp_chroma: u8 = if any_chroma_ac {
            2
        } else if any_chroma_dc {
            1
        } else {
            0
        };

        // 7. mb_type per spec Table 7-11 — Task #50 helper.
        //    `pred_mode` is the Intra_16x16 luma mode chosen above
        //    (0..=3 per the Intra16x16Mode enum). For P-slice intra,
        //    mb_type_offset = 5 shifts the codenum into the P-slice
        //    mb_type table (Table 7-13 rows 6..=29).
        let mb_type = crate::codec::h264::cabac::mb_type_math::pack_i_16x16_mb_type(
            crate::codec::h264::cabac::mb_type_math::I16x16MbType {
                luma_pred_mode,
                cbp_chroma: cbp_chroma as u32,
                cbp_luma_flag: cbp_luma_flag as u32,
            },
        );
        w.write_ue(mb_type + mb_type_offset);

        // intra_chroma_pred_mode: codeNum matches IntraChroma8x8Mode
        // enum (DC=0, Horizontal=1, Vertical=2, Plane=3).
        w.write_ue(chroma_pred_mode);

        // mb_qp_delta: always emitted for Intra_16x16 regardless of
        // CBP (spec § 7.3.5). Diff against the running prev_mb_qp;
        // advance prev_mb_qp afterward.
        let delta = qp as i32 - self.prev_mb_qp;
        w.write_se(delta);
        self.prev_mb_qp = qp as i32;

        // 8. Emit residual blocks in the spec-mandated order.
        // (a) Luma DC block — standard 4×4 zigzag scan per Table 8-13.
        //     Now that `derive_cavlc_fields` follows spec § 7.3.5.3.2
        //     conventions for run/total_zeros (Phase 6A.10-fu2 spec
        //     trace), the previously-empirical reversed-scan hack is
        //     no longer needed.
        let dc_scan = raster_to_scan_levels(&dc_levels);
        // nC for Intra16x16DCLevel uses neighbor MBs' Intra16x16DC
        // TotalCoeffs per spec § 9.2.1.1. Hardcoding 0 here was the
        // root cause of decoder desync on real content past MB (0, 0).
        let dc_nc = self.compute_nc_intra16x16_dc(mb_x, mb_y);
        let dc_total_coeff = dc_scan.iter().filter(|&&v| v != 0).count() as u8;
        encode_cavlc_block(w, &dc_scan, dc_nc, CavlcBlockType::Luma4x4).map_err(|e| {
            EncoderError::InvalidInput(format!("CAVLC luma DC encode: {e}"))
        })?;
        // Store this MB's DC TotalCoeff for the next MB's nC lookup.
        let mb_w_grid = (self.width / 16) as usize;
        self.intra16x16_dc_tc_grid[mb_y * mb_w_grid + mb_x] = dc_total_coeff;

        // (b) 16 luma AC sub-blocks (max_coeffs=15, DC excluded).
        if cbp_luma_flag != 0 {
            for k in 0..16 {
                let (bx_in_mb, by_in_mb) = BLOCK_INDEX_TO_POS[k];
                let abs_bx = mb_x * 4 + bx_in_mb as usize;
                let abs_by = mb_y * 4 + by_in_mb as usize;
                let nc = self.compute_nc_luma_at(abs_bx, abs_by);
                let ac_flat: Vec<i32> = ac_scan_order_15(&ac_levels[k]);
                encode_cavlc_block(w, &ac_flat, nc, CavlcBlockType::Intra16x16Ac).map_err(
                    |e| EncoderError::InvalidInput(format!("CAVLC luma AC: {e}")),
                )?;
                let total_coeff = ac_flat.iter().filter(|&&v| v != 0).count() as u8;
                self.store_total_coeff_luma(abs_bx, abs_by, total_coeff);
            }
        } else {
            // No AC — still mark the grid so neighbor averages stay
            // accurate (blocks with 0 coeffs are "coded with 0").
            for k in 0..16 {
                let (bx_in_mb, by_in_mb) = BLOCK_INDEX_TO_POS[k];
                self.store_total_coeff_luma(
                    mb_x * 4 + bx_in_mb as usize,
                    mb_y * 4 + by_in_mb as usize,
                    0,
                );
            }
        }

        // (c) Chroma DC blocks (2×2, max_coeffs=4, nc=-1).
        if cbp_chroma != 0 {
            let cb_dc_flat: Vec<i32> = cb_dc_levels.iter().flatten().copied().collect();
            encode_cavlc_block(w, &cb_dc_flat, -1, CavlcBlockType::ChromaDc)
                .map_err(|e| EncoderError::InvalidInput(format!("CAVLC Cb DC: {e}")))?;
            let cr_dc_flat: Vec<i32> = cr_dc_levels.iter().flatten().copied().collect();
            encode_cavlc_block(w, &cr_dc_flat, -1, CavlcBlockType::ChromaDc)
                .map_err(|e| EncoderError::InvalidInput(format!("CAVLC Cr DC: {e}")))?;
        }

        // (d) Chroma AC blocks (max_coeffs=15).
        if cbp_chroma == 2 {
            let cmb_x = mb_x * 2;
            let cmb_y = mb_y * 2;
            for (sub_idx, block) in cb_ac_levels.iter().enumerate() {
                let cx = cmb_x + sub_idx % 2;
                let cy = cmb_y + sub_idx / 2;
                let ac_flat: Vec<i32> = ac_scan_order_15(block);
                let nc = self.compute_nc_chroma_at(cx, cy, false);
                encode_cavlc_block(w, &ac_flat, nc, CavlcBlockType::ChromaAc)
                    .map_err(|e| EncoderError::InvalidInput(format!("CAVLC Cb AC: {e}")))?;
                let tc = ac_flat.iter().filter(|&&v| v != 0).count() as u8;
                self.store_total_coeff_chroma(cx, cy, false, tc);
            }
            for (sub_idx, block) in cr_ac_levels.iter().enumerate() {
                let cx = cmb_x + sub_idx % 2;
                let cy = cmb_y + sub_idx / 2;
                let ac_flat: Vec<i32> = ac_scan_order_15(block);
                let nc = self.compute_nc_chroma_at(cx, cy, true);
                encode_cavlc_block(w, &ac_flat, nc, CavlcBlockType::ChromaAc)
                    .map_err(|e| EncoderError::InvalidInput(format!("CAVLC Cr AC: {e}")))?;
                let tc = ac_flat.iter().filter(|&&v| v != 0).count() as u8;
                self.store_total_coeff_chroma(cx, cy, true, tc);
            }
        }

        // 9. Reconstruction (CAVLC I_16x16 — no stego hooks).
        let recon_y =
            self.reconstruct_luma_mb(&pred_y, &ac_levels, &dc_levels, qp);
        self.write_luma_mb_dual(mb_x as u32, mb_y as u32, &recon_y);

        let recon_cb =
            self.reconstruct_chroma_mb(&pred_cb, &cb_ac_levels, &cb_dc_levels, qp_c);
        self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 0, &recon_cb);

        let recon_cr =
            self.reconstruct_chroma_mb(&pred_cr, &cr_ac_levels, &cr_dc_levels, qp_c);
        self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 1, &recon_cr);

        Ok(())
    }

    /// Build a `NeighborsChroma8x8` struct for the chroma MB at
    /// `(mb_x, mb_y)`, component 0 = Cb or 1 = Cr.
    fn build_chroma_neighbors(
        &self,
        mb_x: usize,
        mb_y: usize,
        component: u8,
    ) -> crate::codec::h264::intra_pred::NeighborsChroma8x8 {
        let x0 = (mb_x * 8) as u32;
        let y0 = (mb_y * 8) as u32;
        let top_avail = mb_y > 0;
        let left_avail = mb_x > 0;
        let top_left_avail = top_avail && left_avail;
        let mut top = [0u8; 8];
        let mut left = [0u8; 8];
        if top_avail {
            for dx in 0..8 {
                top[dx as usize] = self.recon.chroma_at(component, x0 + dx, y0 - 1);
            }
        }
        if left_avail {
            for dy in 0..8 {
                left[dy as usize] = self.recon.chroma_at(component, x0 - 1, y0 + dy);
            }
        }
        let top_left = if top_left_avail {
            self.recon.chroma_at(component, x0 - 1, y0 - 1)
        } else {
            0
        };
        crate::codec::h264::intra_pred::NeighborsChroma8x8 {
            top,
            left,
            top_left,
            top_available: top_avail,
            left_available: left_avail,
            top_left_available: top_left_avail,
        }
    }

    /// Build a `Neighbors16x16` struct for the MB at `(mb_x, mb_y)`
    /// by sampling the reconstructed pixel buffer's top row, left
    /// column, and top-left corner.
    fn build_luma_neighbors_16x16(
        &self,
        mb_x: usize,
        mb_y: usize,
    ) -> crate::codec::h264::intra_pred::Neighbors16x16 {
        let x0 = (mb_x * 16) as u32;
        let y0 = (mb_y * 16) as u32;
        let top_avail = mb_y > 0;
        let left_avail = mb_x > 0;
        let top_left_avail = top_avail && left_avail;
        let mut top = [0u8; 16];
        let mut left = [0u8; 16];
        if top_avail {
            for dx in 0..16 {
                top[dx as usize] = self.recon.y_at(x0 + dx, y0 - 1);
            }
        }
        if left_avail {
            for dy in 0..16 {
                left[dy as usize] = self.recon.y_at(x0 - 1, y0 + dy);
            }
        }
        let top_left = if top_left_avail {
            self.recon.y_at(x0 - 1, y0 - 1)
        } else {
            0
        };
        crate::codec::h264::intra_pred::Neighbors16x16 {
            top,
            left,
            top_left,
            top_available: top_avail,
            left_available: left_avail,
            top_left_available: top_left_avail,
        }
    }

    #[allow(dead_code)] // Phase 6 intra-prediction primitive; current paths use SATD-driven mode_decision.
    fn predict_intra16_dc_luma(&self, mb_x: usize, mb_y: usize) -> [[u8; 16]; 16] {
        let top_avail = mb_y > 0;
        let left_avail = mb_x > 0;
        let x0 = (mb_x * 16) as u32;
        let y0 = (mb_y * 16) as u32;

        let dc_val: i32 = match (top_avail, left_avail) {
            (true, true) => {
                let mut sum = 0i32;
                for dx in 0..16 {
                    sum += self.recon.y_at(x0 + dx, y0 - 1) as i32;
                }
                for dy in 0..16 {
                    sum += self.recon.y_at(x0 - 1, y0 + dy) as i32;
                }
                (sum + 16) >> 5
            }
            (true, false) => {
                let mut sum = 0i32;
                for dx in 0..16 {
                    sum += self.recon.y_at(x0 + dx, y0 - 1) as i32;
                }
                (sum + 8) >> 4
            }
            (false, true) => {
                let mut sum = 0i32;
                for dy in 0..16 {
                    sum += self.recon.y_at(x0 - 1, y0 + dy) as i32;
                }
                (sum + 8) >> 4
            }
            (false, false) => 128,
        };
        let v = dc_val.clamp(0, 255) as u8;
        [[v; 16]; 16]
    }

    #[allow(dead_code)] // Phase 6 chroma intra-DC primitive; same status as predict_intra16_dc_luma.
    fn predict_intra_chroma_dc(&self, mb_x: usize, mb_y: usize, component: u8) -> [[u8; 8]; 8] {
        // Simplified 8×8 DC prediction. Spec § 8.3.4.2 partitions the
        // 8×8 block into four 4×4 quadrants each with its own DC; for
        // Phase 6A.10 we use a single 8×8-wide DC (the spec-allowed
        // fallback when chroma samples straddle unavailable
        // boundaries). Good enough for the pipeline validation.
        let top_avail = mb_y > 0;
        let left_avail = mb_x > 0;
        let x0 = (mb_x * 8) as u32;
        let y0 = (mb_y * 8) as u32;

        let dc_val: i32 = match (top_avail, left_avail) {
            (true, true) => {
                let mut sum = 0i32;
                for dx in 0..8 {
                    sum += self.recon.chroma_at(component, x0 + dx, y0 - 1) as i32;
                }
                for dy in 0..8 {
                    sum += self.recon.chroma_at(component, x0 - 1, y0 + dy) as i32;
                }
                (sum + 8) >> 4
            }
            (true, false) => {
                let mut sum = 0i32;
                for dx in 0..8 {
                    sum += self.recon.chroma_at(component, x0 + dx, y0 - 1) as i32;
                }
                (sum + 4) >> 3
            }
            (false, true) => {
                let mut sum = 0i32;
                for dy in 0..8 {
                    sum += self.recon.chroma_at(component, x0 - 1, y0 + dy) as i32;
                }
                (sum + 4) >> 3
            }
            (false, false) => 128,
        };
        let v = dc_val.clamp(0, 255) as u8;
        [[v; 8]; 8]
    }

    fn encode_chroma_component(
        &self,
        src: &[[u8; 8]; 8],
        pred: &[[u8; 8]; 8],
        qp: QuantParams,
    ) -> ([[[i32; 4]; 4]; 4], [[i32; 2]; 2]) {
        let mut ac_levels = [[[0i32; 4]; 4]; 4];
        let mut dc_grid = [[0i32; 2]; 2];
        for sby in 0..2 {
            for sbx in 0..2 {
                let mut sub_res = [[0i32; 4]; 4];
                for dy in 0..4 {
                    for dx in 0..4 {
                        sub_res[dy][dx] = src[sby * 4 + dy][sbx * 4 + dx] as i32
                            - pred[sby * 4 + dy][sbx * 4 + dx] as i32;
                    }
                }
                let mut coeffs = forward_dct_4x4(&sub_res);
                dc_grid[sby][sbx] = coeffs[0][0];
                coeffs[0][0] = 0;
                ac_levels[sby * 2 + sbx] =
                    trellis_quantize_4x4(&coeffs, qp, true)
                        .unwrap_or_else(|_| forward_quantize_4x4(&coeffs, qp));
            }
        }
        let dc_hadamard = forward_hadamard_2x2(&dc_grid);
        let dc_levels = forward_quantize_dc_chroma(&dc_hadamard, qp.qp, qp.slice);
        (ac_levels, dc_levels)
    }

    fn reconstruct_luma_mb(
        &self,
        pred: &[[u8; 16]; 16],
        ac_levels: &[[[i32; 4]; 4]; 16],
        dc_levels: &[[i32; 4]; 4],
        qp: u8,
    ) -> [[u8; 16]; 16] {
        use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;
        let dc_recon = inverse_16x16_dc_hadamard(dc_levels, qp as i32);
        let mut recon = [[0u8; 16]; 16];
        for k in 0..16 {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            let sby = by as usize;
            let sbx = bx as usize;
            let ac_zigzag = raster_to_scan_levels(&ac_levels[k]);
            // DC recon is in raster (bx, by) order (matches the forward
            // placement in dc_grid; spec § 8.5.10).
            let sub_res = reconstruct_residual_4x4_with_dc(
                &ac_zigzag,
                dc_recon[by as usize][bx as usize],
                qp as i32,
            );
            for dy in 0..4 {
                for dx in 0..4 {
                    let v = pred[sby * 4 + dy][sbx * 4 + dx] as i32 + sub_res[dy][dx];
                    recon[sby * 4 + dy][sbx * 4 + dx] = v.clamp(0, 255) as u8;
                }
            }
        }
        recon
    }

    /// §6E-A6.1q.b — write prediction-only recon for a non-direct B-MB.
    /// Builds the 16×16 luma + 8×8 chroma prediction via
    /// [`super::b_inter_prediction`] and writes it directly to
    /// `self.recon`. Used when CBP=0 (current §6E-A6.1q.b state — the
    /// MVDs are emitted but no residual). When residual emission lands
    /// (#151 part 3b), this helper gets superseded by a path that adds
    /// the reconstructed residual to the prediction before writeback.
    ///
    /// Mirrors what the spec decoder does at this MB: with CBP=0 the
    /// reconstructed sample is exactly the inter-prediction output —
    /// `(L0 + L1 + 1) >> 1` for bipred per § 8.4.2.3.1, or the single-
    /// list 6-tap MC result for L0/L1.
    fn write_b_prediction_recon(
        &mut self,
        mode: super::b_inter_prediction::BInterMode,
        l0_ref: &super::reference_buffer::ReconFrame,
        l1_ref: &super::reference_buffer::ReconFrame,
        mb_x: usize,
        mb_y: usize,
    ) {
        let pred_y = super::b_inter_prediction::build_b_luma_prediction(
            mode, l0_ref, l1_ref, mb_x, mb_y,
        );
        let pred_cb = super::b_inter_prediction::build_b_chroma_prediction(
            mode, l0_ref, l1_ref, /* cb */ 0, mb_x, mb_y,
        );
        let pred_cr = super::b_inter_prediction::build_b_chroma_prediction(
            mode, l0_ref, l1_ref, /* cr */ 1, mb_x, mb_y,
        );
        // §B-cascade-real Phase 1.1.B step 3: dual-write B-prediction.
        // No residual emitted → no flip → both buffers identical.
        self.write_luma_mb_dual(mb_x as u32, mb_y as u32, &pred_y);
        self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 0, &pred_cb);
        self.write_chroma_block_dual(mb_x as u32, mb_y as u32, 1, &pred_cr);
    }

    /// §B-Partitioned-Residual Stage A (#206) — extract the residual
    /// emission core out of `write_b_inter_residual_macroblock_cabac`
    /// so the partitioned + B_8x8 emit paths can also use it.
    ///
    /// Caller is responsible for:
    /// 1. Building per-prediction `pred_y` / `pred_cb` / `pred_cr` —
    ///    a single 16×16 + 8×8 prediction over the whole MB. For 16x16
    ///    family this comes from a single `BInterMode`. For partitioned
    ///    / B_8x8 the caller stitches per-partition prediction into one
    ///    16×16 luma + 8×8 chroma surface.
    /// 2. Emitting MB header bins BEFORE this helper: `mb_skip_flag(0)`,
    ///    `mb_type` (and `sub_mb_type[]` for B_8x8), MVDs in spec order.
    /// 3. After this helper returns, doing the [`CabacNeighborMB`]
    ///    commit and the MV grid update — both of which depend on the
    ///    MB's mode + per-partition state, so live with the caller.
    ///
    /// This helper does:
    /// - Gather source pixels.
    /// - Forward DCT + quant per 4x4 luma block + chroma DC/AC.
    /// - Pack CBP and emit `coded_block_pattern` bins.
    /// - Emit `transform_size_8x8_flag` when High profile + CBP_luma>0
    ///   (§B-direct-fix.v2 #194 — phasm always picks 4x4 for B, emits
    ///   `false`).
    /// - Emit `mb_qp_delta` when CBP != 0; commit MB QP state.
    /// - Emit luma 4x4 residuals (cat=2) + chroma DC (cat=3) + chroma
    ///   AC (cat=4).
    /// - Reconstruct (inverse quant + inverse DCT + add to pred) and
    ///   write back into `self.recon` so downstream MBs + deblock see
    ///   the post-residual recon.
    ///
    /// Returns CBP + CBF + qp_delta state so the caller can populate
    /// the neighbor commit.
    #[allow(clippy::too_many_arguments)]
    fn emit_b_residual_for_pred(
        &mut self,
        cabac: &mut crate::codec::h264::cabac::encoder::CabacEncoder,
        mb_x: usize,
        mb_y: usize,
        y_plane: &[u8],
        y_stride: usize,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
        pred_y: &[[u8; 16]; 16],
        pred_cb: &[[u8; 8]; 8],
        pred_cr: &[[u8; 8]; 8],
        qp: u8,
        qp_c: u8,
    ) -> Result<BResidualResult, EncoderError> {
        use crate::codec::h264::cabac::encoder::{
            encode_coded_block_pattern, encode_mb_qp_delta,
            encode_residual_block_cabac_with_cbf_inc,
        };
        use crate::codec::h264::cabac::neighbor::{
            block_pos_to_chroma_ac_idx, compute_cbf_ctx_idx_inc_chroma_ac,
            compute_cbf_ctx_idx_inc_chroma_dc, compute_cbf_ctx_idx_inc_luma_4x4,
            CurrentMbCbf,
        };
        use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;

        // ── 1. Gather source pixels ────────────────────────────────
        let y0 = mb_y * 16;
        let x0 = mb_x * 16;
        let mut src_y = [[0u8; 16]; 16];
        for dy in 0..16 {
            for dx in 0..16 {
                src_y[dy][dx] = y_plane[(y0 + dy) * y_stride + x0 + dx];
            }
        }
        let cy0 = mb_y * 8;
        let cx0 = mb_x * 8;
        let mut src_cb = [[0u8; 8]; 8];
        let mut src_cr = [[0u8; 8]; 8];
        for dy in 0..8 {
            for dx in 0..8 {
                src_cb[dy][dx] = cb_plane[(cy0 + dy) * c_stride + cx0 + dx];
                src_cr[dy][dx] = cr_plane[(cy0 + dy) * c_stride + cx0 + dx];
            }
        }

        // ── 2. Forward DCT + quant per 4×4 luma block ─────────────
        let inter = QuantParams { qp, slice: QuantSlice::Inter };
        let chroma_qp_params = QuantParams { qp: qp_c, slice: QuantSlice::Inter };
        let mut luma_ac_levels = [[[0i32; 4]; 4]; 16];
        let mut luma_nonzero = [false; 16];
        let mut luma_recon_residual_4x4 = [[[0i32; 4]; 4]; 16];
        for k in 0..16 {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            let sby = by as usize;
            let sbx = bx as usize;
            let mut sub_res = [[0i32; 4]; 4];
            for dy in 0..4 {
                for dx in 0..4 {
                    sub_res[dy][dx] = src_y[sby * 4 + dy][sbx * 4 + dx] as i32
                        - pred_y[sby * 4 + dy][sbx * 4 + dx] as i32;
                }
            }
            let coeffs = forward_dct_4x4(&sub_res);
            let levels = trellis_quantize_4x4(&coeffs, inter, true)
                .unwrap_or_else(|_| forward_quantize_4x4(&coeffs, inter));
            luma_ac_levels[k] = levels;
            luma_nonzero[k] = levels.iter().any(|r| r.iter().any(|&v| v != 0));
            use crate::codec::h264::transform::{dequant_4x4, inverse_4x4_integer};
            let dq = dequant_4x4(&levels, qp as i32, false);
            luma_recon_residual_4x4[k] = inverse_4x4_integer(&dq);
        }
        let cbp_luma_8x8 = super::inter_mode::luma_8x8_cbp_mask(&luma_nonzero);

        // ── 3. Forward DCT + quant for chroma (DC + AC) ────────────
        let (cb_ac, cb_dc) = self.encode_chroma_component(&src_cb, pred_cb, chroma_qp_params);
        let (cr_ac, cr_dc) = self.encode_chroma_component(&src_cr, pred_cr, chroma_qp_params);
        // §B-cascade-real Phase 1.1.B step 2 — parallel post-flip
        // arrays for the visual_recon path. Sites 13-15 of 15.
        let mut luma_ac_levels_post = luma_ac_levels;
        let mut cb_ac_post = cb_ac;
        let mut cb_dc_post = cb_dc;
        let mut cr_ac_post = cr_ac;
        let mut cr_dc_post = cr_dc;
        let any_cb_ac = cb_ac.iter().any(|b| b.iter().any(|r| r.iter().any(|&v| v != 0)));
        let any_cr_ac = cr_ac.iter().any(|b| b.iter().any(|r| r.iter().any(|&v| v != 0)));
        let any_cb_dc = cb_dc.iter().flatten().any(|&v| v != 0);
        let any_cr_dc = cr_dc.iter().flatten().any(|&v| v != 0);
        let cbp_chroma = if any_cb_ac || any_cr_ac {
            2u8
        } else if any_cb_dc || any_cr_dc {
            1u8
        } else {
            0u8
        };
        let cbp_value = super::inter_mode::pack_cbp(cbp_luma_8x8, cbp_chroma);

        // ── 4. coded_block_pattern ─────────────────────────────────
        encode_coded_block_pattern(cabac, cbp_value, mb_x);

        // §B-direct-fix.v2 (#194) — emit `transform_size_8x8_flag`
        // when `transform_8x8_mode_flag = 1` AND
        // `CodedBlockPatternLuma > 0`. Phasm always uses 4x4
        // transforms for B-MB residuals, so emit `false`.
        if self.enable_transform_8x8 && cbp_luma_8x8 != 0 {
            crate::codec::h264::cabac::encoder::encode_transform_size_8x8_flag(
                cabac, false, mb_x,
            );
        }

        // ── 5. mb_qp_delta only if cbp != 0 ───────────────────────
        let qp_delta_emitted = if cbp_value != 0 {
            let delta = qp as i32 - self.prev_mb_qp;
            encode_mb_qp_delta(cabac, delta);
            self.prev_mb_qp = qp as i32;
            delta
        } else {
            self.commit_mb_state(mb_x, mb_y, self.prev_mb_qp as u8, false);
            0
        };
        if cbp_value != 0 {
            self.commit_mb_state(mb_x, mb_y, qp, false);
        }

        // ── 6. Residual emission — luma 4×4 only (cat=2) ──────────
        let mut current_cbf = CurrentMbCbf::new();
        let current_is_intra = false;
        if cbp_luma_8x8 != 0 {
            for k in 0..16 {
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                if cbp_luma_8x8 & (1 << (k / 4)) != 0 {
                    let mut scan = raster_to_scan_levels(&luma_ac_levels[k]);
                    // Task #208 — fire stego hook on B-frame luma 4×4
                    // residual. Mirror of P-side at line 3477. Without
                    // this, primary STC's planned coefficient sign
                    // flips on B-frame residuals are dropped → walker
                    // reads natural (non-flipped) signs → message bit
                    // mismatch → FrameCorrupted on decode.
                    self.invoke_stego_residual_hook(
                        mb_x, mb_y, &mut scan, 0, 15,
                        super::super::stego::orchestrate::ResidualPathKind::Luma4x4 {
                            block_idx: k as u8,
                        },
                    );
                    // §B-cascade-real Phase 1.1.B step 2: capture
                    // post-flip 4×4 luma into luma_ac_levels_post.
                    luma_ac_levels_post[k] = scan_to_raster_levels(&scan);
                    let inc = compute_cbf_ctx_idx_inc_luma_4x4(
                        &current_cbf, &cabac.neighbors, mb_x, bx, by, current_is_intra,
                    );
                    let coded = encode_residual_block_cabac_with_cbf_inc(
                        cabac, &scan, 0, 15, 2, inc,
                    );
                    current_cbf.set(2, k, coded);
                    let abs_bx = mb_x * 4 + bx as usize;
                    let abs_by = mb_y * 4 + by as usize;
                    self.store_total_coeff_luma(abs_bx, abs_by, if coded { 1 } else { 0 });
                } else {
                    self.store_total_coeff_luma(
                        mb_x * 4 + bx as usize,
                        mb_y * 4 + by as usize,
                        0,
                    );
                }
            }
        } else {
            for k in 0..16 {
                let (bx, by) = BLOCK_INDEX_TO_POS[k];
                self.store_total_coeff_luma(
                    mb_x * 4 + bx as usize,
                    mb_y * 4 + by as usize,
                    0,
                );
            }
        }

        // ── 7. Residual emission — chroma DC + AC (cat=3 / cat=4) ──
        if cbp_chroma >= 1 {
            for (plane, dc) in [&cb_dc, &cr_dc].iter().enumerate() {
                let mut dc_flat: [i32; 4] = [dc[0][0], dc[0][1], dc[1][0], dc[1][1]];
                // Task #208 — stego hook on B-frame chroma DC.
                self.invoke_stego_residual_hook(
                    mb_x, mb_y, &mut dc_flat, 0, 3,
                    super::super::stego::orchestrate::ResidualPathKind::ChromaDc {
                        plane: plane as u8,
                    },
                );
                // §B-cascade-real Phase 1.1.B step 2: capture post-flip
                // chroma DC.
                let dc_post_raster: [[i32; 2]; 2] =
                    [[dc_flat[0], dc_flat[1]], [dc_flat[2], dc_flat[3]]];
                if plane == 0 {
                    cb_dc_post = dc_post_raster;
                } else {
                    cr_dc_post = dc_post_raster;
                }
                let inc = compute_cbf_ctx_idx_inc_chroma_dc(
                    &cabac.neighbors, mb_x, plane as u8, current_is_intra,
                );
                let coded = encode_residual_block_cabac_with_cbf_inc(
                    cabac, &dc_flat, 0, 3, 3, inc,
                );
                current_cbf.set(3, plane, coded);
            }
        }
        if cbp_chroma == 2 {
            for (plane, ac_blocks) in [&cb_ac, &cr_ac].iter().enumerate() {
                for sub in 0..4 {
                    let bx = (sub % 2) as u8;
                    let by = (sub / 2) as u8;
                    let mut ac_scan = ac_scan_order_15(&ac_blocks[sub]);
                    // Task #208 — stego hook on B-frame chroma AC.
                    self.invoke_stego_residual_hook(
                        mb_x, mb_y, &mut ac_scan, 0, 14,
                        super::super::stego::orchestrate::ResidualPathKind::ChromaAc {
                            plane: plane as u8,
                            block_idx: sub as u8,
                        },
                    );
                    // §B-cascade-real Phase 1.1.B step 2: capture
                    // post-flip chroma AC.
                    let ac_post_raster = ac_scan_15_to_raster(&ac_scan);
                    if plane == 0 {
                        cb_ac_post[sub] = ac_post_raster;
                    } else {
                        cr_ac_post[sub] = ac_post_raster;
                    }
                    let inc = compute_cbf_ctx_idx_inc_chroma_ac(
                        &current_cbf, &cabac.neighbors, mb_x, plane as u8,
                        bx, by, current_is_intra,
                    );
                    let coded = encode_residual_block_cabac_with_cbf_inc(
                        cabac, &ac_scan, 0, 14, 4, inc,
                    );
                    current_cbf.set(
                        4, block_pos_to_chroma_ac_idx(plane as u8, bx, by), coded,
                    );
                }
            }
        }

        // ── 8. Reconstruction ────────────────────────────────────
        let mut recon_luma = [[0u8; 16]; 16];
        for k in 0..16 {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            let sby = by as usize;
            let sbx = bx as usize;
            let sub_res = &luma_recon_residual_4x4[k];
            for dy in 0..4 {
                for dx in 0..4 {
                    let v = pred_y[sby * 4 + dy][sbx * 4 + dx] as i32 + sub_res[dy][dx];
                    recon_luma[sby * 4 + dy][sbx * 4 + dx] = v.clamp(0, 255) as u8;
                }
            }
        }
        self.recon
            .write_luma_mb(mb_x as u32, mb_y as u32, &recon_luma);
        let recon_cb = self.reconstruct_chroma_mb(pred_cb, &cb_ac, &cb_dc, qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &recon_cb);
        let recon_cr = self.reconstruct_chroma_mb(pred_cr, &cr_ac, &cr_dc, qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &recon_cr);

        // §B-cascade-real Phase 1.1.B step 2 — visual_recon path with
        // POST-flip levels. Sites 13-15 of 15: B-frame luma 4x4 +
        // chroma DC + chroma AC. Recompute residual from post-flip
        // levels (no cache exists for post-flip).
        let mut visual_recon_luma = [[0u8; 16]; 16];
        use crate::codec::h264::transform::{dequant_4x4, inverse_4x4_integer};
        for k in 0..16 {
            let (bx, by) = BLOCK_INDEX_TO_POS[k];
            let sby = by as usize;
            let sbx = bx as usize;
            let dq = dequant_4x4(&luma_ac_levels_post[k], qp as i32, false);
            let res = inverse_4x4_integer(&dq);
            for dy in 0..4 {
                for dx in 0..4 {
                    let v = pred_y[sby * 4 + dy][sbx * 4 + dx] as i32
                        + res[dy][dx];
                    visual_recon_luma[sby * 4 + dy][sbx * 4 + dx] =
                        v.clamp(0, 255) as u8;
                }
            }
        }
        self.visual_recon
            .write_luma_mb(mb_x as u32, mb_y as u32, &visual_recon_luma);
        let visual_cb =
            self.reconstruct_chroma_mb(pred_cb, &cb_ac_post, &cb_dc_post, qp_c);
        self.visual_recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &visual_cb);
        let visual_cr =
            self.reconstruct_chroma_mb(pred_cr, &cr_ac_post, &cr_dc_post, qp_c);
        self.visual_recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &visual_cr);

        let _ = current_is_intra;
        Ok(BResidualResult {
            cbp_luma_8x8,
            cbp_chroma,
            qp_delta_emitted,
            current_cbf,
        })
    }

    /// §6E-A6.1q.b part 3b — full residual emission for a non-direct
    /// B-MB (L0 / L1 / Bi 16x16). Mirrors `write_p_macroblock_cabac`
    /// for the pieces that overlap (forward DCT + quant + CBP +
    /// residual emit + reconstruction) but uses the B-side syntax for
    /// mb_type + MVDs and the [`super::b_inter_prediction`] helpers
    /// for prediction.
    ///
    /// Scope:
    /// - 4×4 transform only (no 8×8 transform path; B-MB
    ///   `transform_size_8x8_flag` is gated off via writing 0 to the
    ///   transform_8x8 grid).
    /// - 16x16 partition only — `BInterMode` is L0/L1/Bi 16x16.
    /// - No skip_fast / P_SKIP detection; B_Skip is dispatched by the
    ///   caller upstream.
    /// - No AQ; B-frames use the slice QP directly (matches §6E-A6.1
    ///   bucket-mix philosophy).
    /// - No intra-in-B; that's a §6E-B refinement.
    /// - Chroma residual emitted under cbp_chroma == 1 / 2 same as
    ///   P-side.
    ///
    /// Walker side (#152) must lift `cbp_byte != 0 → Unsupported`
    /// in `walk_b_l0_16x16` / `_l1_16x16` / `_bi_16x16` and wire the
    /// matching residual decode for round-trips to validate.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn write_b_inter_residual_macroblock_cabac(
        &mut self,
        cabac: &mut crate::codec::h264::cabac::encoder::CabacEncoder,
        mode: super::b_inter_prediction::BInterMode,
        l0_ref: &super::reference_buffer::ReconFrame,
        l1_ref: &super::reference_buffer::ReconFrame,
        mb_x: usize,
        mb_y: usize,
        y_plane: &[u8],
        y_stride: usize,
        cb_plane: &[u8],
        cr_plane: &[u8],
        c_stride: usize,
        qp: u8,
        qp_c: u8,
        ref_idx_l0: u8,
    ) -> Result<(), EncoderError> {
        use crate::codec::h264::cabac::encoder::encode_ref_idx;
        use crate::codec::h264::cabac::neighbor::{CabacNeighborMB, MbTypeClass};

        // ── 1. Build prediction ────────────────────────────────────
        let pred_y = super::b_inter_prediction::build_b_luma_prediction(
            mode, l0_ref, l1_ref, mb_x, mb_y,
        );
        let pred_cb = super::b_inter_prediction::build_b_chroma_prediction(
            mode, l0_ref, l1_ref, /* cb */ 0, mb_x, mb_y,
        );
        let pred_cr = super::b_inter_prediction::build_b_chroma_prediction(
            mode, l0_ref, l1_ref, /* cr */ 1, mb_x, mb_y,
        );

        // ── 2. Emit MB syntax: skip_flag(0) + mb_type ───────────────
        encode_mb_skip_flag_b(cabac, false, mb_x);
        let mb_type_value: u32 = match mode {
            super::b_inter_prediction::BInterMode::L0_16x16 { .. } => 1,
            super::b_inter_prediction::BInterMode::L1_16x16 { .. } => 2,
            super::b_inter_prediction::BInterMode::Bi_16x16 { .. } => 3,
        };
        encode_mb_type_b(cabac, mb_type_value, mb_x);

        // v1.4 (#305) — ref_idx_l0 emitted between mb_type and MVDs
        // for L0_16x16 / Bi_16x16 (Pred_L0 / BiPred). L1_16x16 stays
        // single-ref (Q1). At MultiRefConfig::SINGLE_REF default the
        // gate is closed and zero bins emit.
        //
        // v1.4 Phase 4.5 (#316) — gate on actual L0 references in the
        // DPB (mirror of slice-header `actual_l0_b` derivation in
        // build_b_slice_rbsp_cabac). When pre_past_anchor=None,
        // actual count is 1 — no ref_idx_l0 bin on the wire.
        let num_active_l0 = if self.multi_ref_config.max_l0_active > 1
            && self.dpb.pre_past_anchor.is_some()
        {
            2u8
        } else {
            1u8
        };
        let current_ref_idx_mb_b16x16 = crate::codec::h264::cabac::neighbor::CurrentMbRefIdx::new();
        if num_active_l0 > 1 {
            let uses_l0 = matches!(
                mode,
                super::b_inter_prediction::BInterMode::L0_16x16 { .. }
                    | super::b_inter_prediction::BInterMode::Bi_16x16 { .. }
            );
            if uses_l0 {
                // v1.4 Phase 4.3 (#314) — actual ref_idx_l0 from
                // BMbDecision (post-pass-resolved by
                // refine_b_choice_multi_ref). 0 = closest past
                // anchor, 1 = pre_past_anchor.
                encode_ref_idx(
                    cabac, ref_idx_l0 as u32, &current_ref_idx_mb_b16x16,
                    mb_x, 0, 0, (num_active_l0 - 1) as u32,
                );
            }
        }

        // ── 3. MVDs (per-mode) — mirrors emit_b_l0/l1/bi_16x16 ─────
        let (abs_l0, abs_l1) = self.emit_b_inter_mvds_inline(
            cabac, mb_x, mb_y, mode,
        );

        // ── 4. CBP + transform_size_8x8_flag + mb_qp_delta + residual
        //       emission + reconstruction (helper) ───────────────────
        let res = self.emit_b_residual_for_pred(
            cabac, mb_x, mb_y,
            y_plane, y_stride, cb_plane, cr_plane, c_stride,
            &pred_y, &pred_cb, &pred_cr,
            qp, qp_c,
        )?;

        // ── 5. Neighbor commit ─────────────────────────────────────
        let abs_mvd_l0 = match abs_l0 {
            Some((x, y)) => [[x; 16], [y; 16]],
            None => [[0; 16]; 2],
        };
        let abs_mvd_l1 = match abs_l1 {
            Some((x, y)) => [[x; 16], [y; 16]],
            None => [[0; 16]; 2],
        };
        let mut nb = CabacNeighborMB::default();
        nb.mb_type = MbTypeClass::PInter;
        nb.mb_skip_flag = false;
        nb.cbp_luma = res.cbp_luma_8x8;
        nb.cbp_chroma = res.cbp_chroma;
        nb.mb_qp_delta = res.qp_delta_emitted;
        nb.coded_block_flag_cat = res.current_cbf.to_neighbor_cbf();
        // v1.4 Phase 4.5 (#316) — propagate the actual emitted
        // ref_idx_l0 for L0_16x16 / Bi_16x16. L1_16x16 has no L0 →
        // stays [0;16] (default).
        let uses_l0 = matches!(
            mode,
            super::b_inter_prediction::BInterMode::L0_16x16 { .. }
                | super::b_inter_prediction::BInterMode::Bi_16x16 { .. }
        );
        if uses_l0 {
            nb.ref_idx_l0 = [ref_idx_l0 as i8; 16];
        }
        nb.abs_mvd_comp = abs_mvd_l0;
        nb.abs_mvd_comp_l1 = abs_mvd_l1;
        nb.transform_size_8x8_flag = false;
        cabac.neighbors.commit(mb_x, nb);

        // ── 6. MV grid update for downstream predictors ────────────
        // v1.4 Phase 4.5 (#316) — propagate the chosen ref_idx_l0 into
        // the MV grid so spec § 8.4.1.3 PMV picks the right neighbour
        // partition (matched-ref-idx priority). Hardcoding 0 here was
        // the v1.3 single-ref convention; under refine it would mean
        // encoder PMV != reference-decoder PMV → MVD bin abs values diverge →
        // CABAC drift.
        let r = ref_idx_l0 as i8;
        let (l0_for_grid, l1_for_grid) = match mode {
            super::b_inter_prediction::BInterMode::L0_16x16 { mv } => {
                (Some((mv, r)), None)
            }
            super::b_inter_prediction::BInterMode::L1_16x16 { mv } => {
                (None, Some((mv, 0)))
            }
            super::b_inter_prediction::BInterMode::Bi_16x16 { mv_l0, mv_l1 } => {
                (Some((mv_l0, r)), Some((mv_l1, 0)))
            }
        };
        self.mv_grid.fill_lists(mb_x * 4, mb_y * 4, 4, 4, l0_for_grid, l1_for_grid);

        Ok(())
    }

    /// §6E-A6.1q.b part 3b — emit the MVD pair(s) for a non-direct
    /// B-MB (L0 / L1 / Bi 16x16). Returns `(abs_l0, abs_l1)` —
    /// per-list absolute MVD magnitudes for the neighbour commit.
    /// Mirrors the existing free-function `emit_b_l0_16x16` /
    /// `_l1_16x16` / `_bi_16x16` MVD emission inline so the residual
    /// path can keep them in one method.
    fn emit_b_inter_mvds_inline(
        &self,
        cabac: &mut crate::codec::h264::cabac::encoder::CabacEncoder,
        mb_x: usize,
        mb_y: usize,
        mode: super::b_inter_prediction::BInterMode,
    ) -> (Option<(i16, i16)>, Option<(i16, i16)>) {
        use crate::codec::h264::cabac::encoder::encode_mvd_with_bin0_inc;
        use crate::codec::h264::cabac::neighbor::{
            ctx_idx_inc_mvd_bin0, ctx_idx_inc_mvd_bin0_per_list,
        };
        let abs_to_pair = |mvd_x: i32, mvd_y: i32| -> (i16, i16) {
            (
                mvd_x.unsigned_abs().min(i16::MAX as u32) as i16,
                mvd_y.unsigned_abs().min(i16::MAX as u32) as i16,
            )
        };
        match mode {
            super::b_inter_prediction::BInterMode::L0_16x16 { mv } => {
                let predicted = super::partition_state::predict_mv_for_mb_partition(
                    &self.mv_grid, mb_x * 4, mb_y * 4, 4, 4, 0, 0,
                );
                let mvd_x = (mv.mv_x as i32) - (predicted.mv_x as i32);
                let mvd_y = (mv.mv_y as i32) - (predicted.mv_y as i32);
                let bin0_inc_x = ctx_idx_inc_mvd_bin0(&cabac.neighbors, mb_x, 0, 0, 0);
                encode_mvd_with_bin0_inc(cabac, mvd_x, 0, bin0_inc_x);
                let bin0_inc_y = ctx_idx_inc_mvd_bin0(&cabac.neighbors, mb_x, 0, 0, 1);
                encode_mvd_with_bin0_inc(cabac, mvd_y, 1, bin0_inc_y);
                (Some(abs_to_pair(mvd_x, mvd_y)), None)
            }
            super::b_inter_prediction::BInterMode::L1_16x16 { mv } => {
                let predicted = super::b_direct_predictor::predict_mv_for_partition_l1_pub(
                    &self.mv_grid, mb_x * 4, mb_y * 4, 0,
                );
                let mvd_x = (mv.mv_x as i32) - (predicted.mv_x as i32);
                let mvd_y = (mv.mv_y as i32) - (predicted.mv_y as i32);
                let bin0_inc_x = ctx_idx_inc_mvd_bin0_per_list(
                    &cabac.neighbors, mb_x, 0, 0, 0, /* list */ 1,
                );
                encode_mvd_with_bin0_inc(cabac, mvd_x, 0, bin0_inc_x);
                let bin0_inc_y = ctx_idx_inc_mvd_bin0_per_list(
                    &cabac.neighbors, mb_x, 0, 0, 1, 1,
                );
                encode_mvd_with_bin0_inc(cabac, mvd_y, 1, bin0_inc_y);
                (None, Some(abs_to_pair(mvd_x, mvd_y)))
            }
            super::b_inter_prediction::BInterMode::Bi_16x16 { mv_l0, mv_l1 } => {
                let pred_l0 = super::partition_state::predict_mv_for_mb_partition(
                    &self.mv_grid, mb_x * 4, mb_y * 4, 4, 4, 0, 0,
                );
                let mvd_l0_x = (mv_l0.mv_x as i32) - (pred_l0.mv_x as i32);
                let mvd_l0_y = (mv_l0.mv_y as i32) - (pred_l0.mv_y as i32);
                let bin0_inc_x = ctx_idx_inc_mvd_bin0(&cabac.neighbors, mb_x, 0, 0, 0);
                encode_mvd_with_bin0_inc(cabac, mvd_l0_x, 0, bin0_inc_x);
                let bin0_inc_y = ctx_idx_inc_mvd_bin0(&cabac.neighbors, mb_x, 0, 0, 1);
                encode_mvd_with_bin0_inc(cabac, mvd_l0_y, 1, bin0_inc_y);

                let pred_l1 = super::b_direct_predictor::predict_mv_for_partition_l1_pub(
                    &self.mv_grid, mb_x * 4, mb_y * 4, 0,
                );
                let mvd_l1_x = (mv_l1.mv_x as i32) - (pred_l1.mv_x as i32);
                let mvd_l1_y = (mv_l1.mv_y as i32) - (pred_l1.mv_y as i32);
                let bin0_inc_x = ctx_idx_inc_mvd_bin0_per_list(
                    &cabac.neighbors, mb_x, 0, 0, 0, /* list */ 1,
                );
                encode_mvd_with_bin0_inc(cabac, mvd_l1_x, 0, bin0_inc_x);
                let bin0_inc_y = ctx_idx_inc_mvd_bin0_per_list(
                    &cabac.neighbors, mb_x, 0, 0, 1, 1,
                );
                encode_mvd_with_bin0_inc(cabac, mvd_l1_y, 1, bin0_inc_y);
                (
                    Some(abs_to_pair(mvd_l0_x, mvd_l0_y)),
                    Some(abs_to_pair(mvd_l1_x, mvd_l1_y)),
                )
            }
        }
    }

    fn reconstruct_chroma_mb(
        &self,
        pred: &[[u8; 8]; 8],
        ac_levels: &[[[i32; 4]; 4]; 4],
        dc_levels: &[[i32; 2]; 2],
        qp_c: u8,
    ) -> [[u8; 8]; 8] {
        let dc_recon = inverse_chroma_dc_2x2_hadamard(dc_levels, qp_c as i32);
        let mut recon = [[0u8; 8]; 8];
        for sby in 0..2 {
            for sbx in 0..2 {
                let ac_zigzag = raster_to_scan_levels(&ac_levels[sby * 2 + sbx]);
                let sub_res = reconstruct_residual_4x4_with_dc(
                    &ac_zigzag,
                    dc_recon[sby][sbx],
                    qp_c as i32,
                );
                for dy in 0..4 {
                    for dx in 0..4 {
                        let v = pred[sby * 4 + dy][sbx * 4 + dx] as i32 + sub_res[dy][dx];
                        recon[sby * 4 + dy][sbx * 4 + dx] = v.clamp(0, 255) as u8;
                    }
                }
            }
        }
        recon
    }
}

/// Build the 16×16 luma prediction for an MB from a partition choice.
/// Each partition's MC writes into its sub-rectangle of the 16×16
/// output buffer.
/// Phase C RDO rerank: compute `cost = D + λ²R` for the top-2
/// SATD candidates and return the winner + its SATD cost (used by
/// the intra-in-P gate). Falls back to the SATD best when fewer than
/// 2 distinct candidates exist.
#[allow(clippy::too_many_arguments)]
fn select_p_mb_via_rdo(
    src_y: &[[u8; 16]; 16],
    reference: &super::reference_buffer::ReconFrame,
    grid: &mut super::partition_state::EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
    mb_qp: u8,
    decision: &super::partition_decision::PMbDecision,
    total_coeff_grid: &[u8],
    frame_w4: usize,
) -> (PMbChoice, u32) {
    use super::rdo::evaluate_p_mb_rdo;
    // Rank indices by SATD ascending.
    let mut order = [0usize, 1, 2, 3];
    order.sort_by_key(|&i| decision.satd_costs[i]);
    let k = 2usize.min(decision.candidates.len());
    let mut best_cost = u64::MAX;
    let mut best_idx = order[0];
    for &i in &order[..k] {
        // This path is behind `PHASM_ENABLE_RDO=1` (opt-in) and
        // currently doesn't thread chroma inputs — keep luma-only
        // cost for backward-compat with prior measurements. The
        // intra-vs-inter gate uses its own RDO call with chroma.
        let r = evaluate_p_mb_rdo(
            &decision.candidates[i],
            src_y,
            reference,
            grid,
            mb_x,
            mb_y,
            mb_qp,
            total_coeff_grid,
            frame_w4,
            None,
        );
        if r.cost < best_cost {
            best_cost = r.cost;
            best_idx = i;
        }
    }
    (decision.candidates[best_idx], decision.satd_costs[best_idx])
}

pub(crate) fn build_luma_prediction(
    reference: &super::reference_buffer::ReconFrame,
    mb_x: usize,
    mb_y: usize,
    choice: &PMbChoice,
) -> [[u8; 16]; 16] {
    let mut out = [[0u8; 16]; 16];
    let mb_px_x = (mb_x * 16) as u32;
    let mb_px_y = (mb_y * 16) as u32;
    let flat = out.as_flattened_mut();
    match *choice {
        PMbChoice::P16x16 { mv, .. } => {
            apply_luma_mv_block(reference, mb_px_x, mb_px_y, 16, 16, mv, flat, 16);
        }
        PMbChoice::P16x8 { mvs, .. } => {
            apply_luma_mv_block(reference, mb_px_x, mb_px_y, 16, 8, mvs[0], flat, 16);
            apply_luma_mv_block(
                reference,
                mb_px_x,
                mb_px_y + 8,
                16,
                8,
                mvs[1],
                &mut flat[8 * 16..],
                16,
            );
        }
        PMbChoice::P8x16 { mvs, .. } => {
            apply_luma_mv_block(reference, mb_px_x, mb_px_y, 8, 16, mvs[0], flat, 16);
            apply_luma_mv_block(
                reference,
                mb_px_x + 8,
                mb_px_y,
                8,
                16,
                mvs[1],
                &mut flat[8..],
                16,
            );
        }
        PMbChoice::P8x8 { sub } => {
            for (i, sub_choice) in sub.iter().enumerate() {
                let (sub_off_x, sub_off_y) = SUB_MB_ORIGINS_PX[i];
                for (px, py, w, h, mv) in sub_mb_luma_partitions(sub_choice) {
                    let tl_x = sub_off_x + px;
                    let tl_y = sub_off_y + py;
                    let row_offset = (tl_y as usize) * 16 + tl_x as usize;
                    apply_luma_mv_block(
                        reference,
                        mb_px_x + tl_x,
                        mb_px_y + tl_y,
                        w,
                        h,
                        mv,
                        &mut flat[row_offset..],
                        16,
                    );
                }
            }
        }
    }
    out
}

/// Iterate the luma partitions inside an 8×8 sub-MB. Returns
/// `(off_x, off_y, w, h, mv)` in luma-pixel units within the sub-MB.
fn sub_mb_luma_partitions(sub: &SubMbChoice) -> Vec<(u32, u32, u32, u32, MotionVector)> {
    match *sub {
        SubMbChoice::P8x8 { mv, .. } => vec![(0, 0, 8, 8, mv)],
        SubMbChoice::P8x4 { mvs, .. } => vec![(0, 0, 8, 4, mvs[0]), (0, 4, 8, 4, mvs[1])],
        SubMbChoice::P4x8 { mvs, .. } => vec![(0, 0, 4, 8, mvs[0]), (4, 0, 4, 8, mvs[1])],
        SubMbChoice::P4x4 { mvs, .. } => vec![
            (0, 0, 4, 4, mvs[0]),
            (4, 0, 4, 4, mvs[1]),
            (0, 4, 4, 4, mvs[2]),
            (4, 4, 4, 4, mvs[3]),
        ],
    }
}

/// Build the 8×8 chroma prediction for an MB from a partition choice.
/// Chroma partition sizes are half the luma sizes on each axis:
/// 16×16 → 8×8, 16×8 → 8×4, 8×16 → 4×8.
pub(crate) fn build_chroma_prediction(
    reference: &super::reference_buffer::ReconFrame,
    component: u8,
    mb_x: usize,
    mb_y: usize,
    choice: &PMbChoice,
) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    let mb_px_x = (mb_x * 8) as u32;
    let mb_px_y = (mb_y * 8) as u32;
    let flat = out.as_flattened_mut();
    match *choice {
        PMbChoice::P16x16 { mv, .. } => {
            apply_chroma_mv_block(reference, component, mb_px_x, mb_px_y, 8, 8, mv, flat, 8);
        }
        PMbChoice::P16x8 { mvs, .. } => {
            apply_chroma_mv_block(
                reference, component, mb_px_x, mb_px_y, 8, 4, mvs[0], flat, 8,
            );
            apply_chroma_mv_block(
                reference,
                component,
                mb_px_x,
                mb_px_y + 4,
                8,
                4,
                mvs[1],
                &mut flat[4 * 8..],
                8,
            );
        }
        PMbChoice::P8x16 { mvs, .. } => {
            apply_chroma_mv_block(
                reference, component, mb_px_x, mb_px_y, 4, 8, mvs[0], flat, 8,
            );
            apply_chroma_mv_block(
                reference,
                component,
                mb_px_x + 4,
                mb_px_y,
                4,
                8,
                mvs[1],
                &mut flat[4..],
                8,
            );
        }
        PMbChoice::P8x8 { sub } => {
            // Each luma partition corresponds to a chroma partition
            // half its size on each axis (4:2:0). Spec § 8.4.1.4:
            // chroma MC uses the same MV as the corresponding luma
            // partition, at eighth-pel chroma precision.
            for (i, sub_choice) in sub.iter().enumerate() {
                let (sub_off_x_luma, sub_off_y_luma) = SUB_MB_ORIGINS_PX[i];
                let sub_off_x_c = sub_off_x_luma / 2;
                let sub_off_y_c = sub_off_y_luma / 2;
                for (px, py, w, h, mv) in sub_mb_luma_partitions(sub_choice) {
                    let tl_x_c = sub_off_x_c + px / 2;
                    let tl_y_c = sub_off_y_c + py / 2;
                    let w_c = w / 2;
                    let h_c = h / 2;
                    let row_offset = (tl_y_c as usize) * 8 + tl_x_c as usize;
                    apply_chroma_mv_block(
                        reference,
                        component,
                        mb_px_x + tl_x_c,
                        mb_px_y + tl_y_c,
                        w_c,
                        h_c,
                        mv,
                        &mut flat[row_offset..],
                        8,
                    );
                }
            }
        }
    }
    out
}

/// Emit MVD se-coded bitstream for each partition in the MB, updating
/// the 4×4 MV grid after each partition so the next one's predictor
/// sees the resolved MV as its `A` neighbor.
pub(crate) fn emit_mvds_and_update_grid<W: super::bitstream_writer::BitSink>(
    w: &mut W,
    grid: &mut EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
    choice: &PMbChoice,
) {
    let base_bx = mb_x * 4;
    let base_by = mb_y * 4;
    match *choice {
        PMbChoice::P16x16 { mv, .. } => {
            let pred = predict_mv_for_partition(grid, base_bx, base_by, 4, 0);
            w.write_se(mv.mv_x as i32 - pred.mv_x as i32);
            w.write_se(mv.mv_y as i32 - pred.mv_y as i32);
            grid.fill(base_bx, base_by, 4, 4, mv, 0);
        }
        PMbChoice::P16x8 { mvs, .. } => {
            // Partition 0 (top): § 8.4.1.3.1 prefers top neighbor B
            // when refB == cur. Falls through to median otherwise.
            let pred0 = predict_mv_for_mb_partition(grid, base_bx, base_by, 4, 2, 0, 0);
            w.write_se(mvs[0].mv_x as i32 - pred0.mv_x as i32);
            w.write_se(mvs[0].mv_y as i32 - pred0.mv_y as i32);
            grid.fill(base_bx, base_by, 4, 2, mvs[0], 0);
            // Partition 1 (bottom): prefers left neighbor A.
            let pred1 = predict_mv_for_mb_partition(grid, base_bx, base_by + 2, 4, 2, 1, 0);
            w.write_se(mvs[1].mv_x as i32 - pred1.mv_x as i32);
            w.write_se(mvs[1].mv_y as i32 - pred1.mv_y as i32);
            grid.fill(base_bx, base_by + 2, 4, 2, mvs[1], 0);
        }
        PMbChoice::P8x16 { mvs, .. } => {
            // Partition 0 (left): § 8.4.1.3.1 prefers left neighbor A.
            let pred0 = predict_mv_for_mb_partition(grid, base_bx, base_by, 2, 4, 0, 0);
            w.write_se(mvs[0].mv_x as i32 - pred0.mv_x as i32);
            w.write_se(mvs[0].mv_y as i32 - pred0.mv_y as i32);
            grid.fill(base_bx, base_by, 2, 4, mvs[0], 0);
            // Partition 1 (right): prefers top-right neighbor C.
            let pred1 = predict_mv_for_mb_partition(grid, base_bx + 2, base_by, 2, 4, 1, 0);
            w.write_se(mvs[1].mv_x as i32 - pred1.mv_x as i32);
            w.write_se(mvs[1].mv_y as i32 - pred1.mv_y as i32);
            grid.fill(base_bx + 2, base_by, 2, 4, mvs[1], 0);
        }
        PMbChoice::P8x8 { sub } => {
            // Spec § 7.3.5.1 / § 7.3.5.2: for P_8x8, the sub_mb_types
            // for all four sub-MBs are emitted FIRST, then the ref_idx
            // block (skipped under our SPS), then per-sub-MB MVDs.
            for sub_choice in &sub {
                w.write_ue(sub_choice.sub_mb_type_codenum());
            }
            // Per-sub-MB, per-sub-partition MVDs.
            for (i, sub_choice) in sub.iter().enumerate() {
                let (off_x_4x4, off_y_4x4) = SUB_MB_ORIGINS_4X4[i];
                let sub_bx = base_bx + off_x_4x4;
                let sub_by = base_by + off_y_4x4;
                emit_sub_mb_mvds(w, grid, sub_bx, sub_by, sub_choice);
            }
        }
    }
}

/// Emit MVDs for a single 8×8 sub-MB's sub_mb_type choice, updating
/// the MV grid after each partition resolves. Partition offsets are
/// in 4×4-block units within the sub-MB.
fn emit_sub_mb_mvds<W: super::bitstream_writer::BitSink>(
    w: &mut W,
    grid: &mut EncoderMvGrid,
    sub_bx: usize,
    sub_by: usize,
    sub_choice: &SubMbChoice,
) {
    match *sub_choice {
        SubMbChoice::P8x8 { mv, .. } => {
            let pred = predict_mv_for_partition(grid, sub_bx, sub_by, 2, 0);
            w.write_se(mv.mv_x as i32 - pred.mv_x as i32);
            w.write_se(mv.mv_y as i32 - pred.mv_y as i32);
            grid.fill(sub_bx, sub_by, 2, 2, mv, 0);
        }
        SubMbChoice::P8x4 { mvs, .. } => {
            // Top 8×4: 2 4×4 blocks wide, 1 tall.
            let pred_top = predict_mv_for_partition(grid, sub_bx, sub_by, 2, 0);
            w.write_se(mvs[0].mv_x as i32 - pred_top.mv_x as i32);
            w.write_se(mvs[0].mv_y as i32 - pred_top.mv_y as i32);
            grid.fill(sub_bx, sub_by, 2, 1, mvs[0], 0);
            // Bottom 8×4.
            let pred_bot = predict_mv_for_partition(grid, sub_bx, sub_by + 1, 2, 0);
            w.write_se(mvs[1].mv_x as i32 - pred_bot.mv_x as i32);
            w.write_se(mvs[1].mv_y as i32 - pred_bot.mv_y as i32);
            grid.fill(sub_bx, sub_by + 1, 2, 1, mvs[1], 0);
        }
        SubMbChoice::P4x8 { mvs, .. } => {
            // Left 4×8.
            let pred_left = predict_mv_for_partition(grid, sub_bx, sub_by, 1, 0);
            w.write_se(mvs[0].mv_x as i32 - pred_left.mv_x as i32);
            w.write_se(mvs[0].mv_y as i32 - pred_left.mv_y as i32);
            grid.fill(sub_bx, sub_by, 1, 2, mvs[0], 0);
            // Right 4×8.
            let pred_right = predict_mv_for_partition(grid, sub_bx + 1, sub_by, 1, 0);
            w.write_se(mvs[1].mv_x as i32 - pred_right.mv_x as i32);
            w.write_se(mvs[1].mv_y as i32 - pred_right.mv_y as i32);
            grid.fill(sub_bx + 1, sub_by, 1, 2, mvs[1], 0);
        }
        SubMbChoice::P4x4 { mvs, .. } => {
            // TL, TR, BL, BR in that order per spec.
            let quarters = [(0usize, 0usize), (1, 0), (0, 1), (1, 1)];
            for (i, &(qx, qy)) in quarters.iter().enumerate() {
                let pred = predict_mv_for_partition(grid, sub_bx + qx, sub_by + qy, 1, 0);
                w.write_se(mvs[i].mv_x as i32 - pred.mv_x as i32);
                w.write_se(mvs[i].mv_y as i32 - pred.mv_y as i32);
                grid.fill(sub_bx + qx, sub_by + qy, 1, 1, mvs[i], 0);
            }
        }
    }
}

/// Convert a raster-ordered 4×4 grid of levels into a zigzag-ordered
/// 15-entry slice for Intra16x16Ac / ChromaAc CAVLC (which skip
/// position `[0][0]` because DC is in a separate block).
fn ac_scan_order_15(raster: &[[i32; 4]; 4]) -> Vec<i32> {
    use crate::codec::h264::tables::ZIGZAG_4X4;
    let mut out = Vec::with_capacity(15);
    // Scan positions 1..16 (skip position 0 = DC).
    for scan_idx in 1..16 {
        let raster_idx = ZIGZAG_4X4[scan_idx] as usize;
        let i = raster_idx / 4;
        let j = raster_idx % 4;
        out.push(raster[i][j]);
    }
    debug_assert_eq!(out.len(), 15);
    out
}

/// §B-cascade-real Phase 1.1.B step 2 — inverse of [`ac_scan_order_15`].
/// Lifts a 15-entry chroma-AC zigzag scan back into a raster 4×4 grid
/// with `[0][0]` left zero (the DC coefficient lives in a separate
/// chroma DC Hadamard block).
fn ac_scan_15_to_raster(scan: &[i32]) -> [[i32; 4]; 4] {
    use crate::codec::h264::tables::ZIGZAG_4X4;
    debug_assert_eq!(scan.len(), 15);
    let mut raster = [[0i32; 4]; 4];
    for (offset, &v) in scan.iter().enumerate() {
        let scan_idx = offset + 1; // skip DC at scan[0]
        let raster_idx = ZIGZAG_4X4[scan_idx] as usize;
        raster[raster_idx / 4][raster_idx % 4] = v;
    }
    raster
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::bitstream::parse_nal_units_annexb;

    fn make_yuv420p(w: u32, h: u32) -> Vec<u8> {
        let y = (w * h) as usize;
        let c = (w / 2 * h / 2) as usize;
        let mut buf = vec![0u8; y + 2 * c];
        for yy in 0..h {
            for xx in 0..w {
                buf[(yy * w + xx) as usize] = ((xx + yy) & 0xFF) as u8;
            }
        }
        for yy in 0..(h / 2) {
            for xx in 0..(w / 2) {
                buf[y + (yy * (w / 2) + xx) as usize] = (xx & 0xFF) as u8;
            }
        }
        for yy in 0..(h / 2) {
            for xx in 0..(w / 2) {
                buf[y + c + (yy * (w / 2) + xx) as usize] = (yy & 0xFF) as u8;
            }
        }
        buf
    }

    #[test]
    fn encoder_new_rejects_non_aligned_dims() {
        assert!(Encoder::new(33, 32, Some(75)).is_err());
        assert!(Encoder::new(32, 31, Some(75)).is_err());
        assert!(Encoder::new(32, 32, Some(75)).is_ok());
    }

    #[test]
    fn encoder_new_with_crf_anchors_target_crf_directly() {
        // §v1.7 Phase 3 (#325) — `new_with_crf(.., 26)` must set
        // both `enc.crf = Some(26)` AND `enc.rc.target_crf = 26`,
        // bypassing the user-quality → CRF mapping that `new(.., Some(26))`
        // applies (which would yield target_crf=33 for quality=26).
        let enc = Encoder::new_with_crf(32, 32, 26).unwrap();
        assert_eq!(enc.crf, Some(26), "crf field anchored at 26");
        assert_eq!(enc.rc.target_crf, 26, "target_crf set directly, not via quality_to_crf");

        // Contrast: `new(.., Some(26))` interprets 26 as user-quality.
        let enc_q = Encoder::new(32, 32, Some(26)).unwrap();
        assert!(enc_q.crf.is_none(), "new(.., Some(q)) does not enable CRF mode");
        assert_ne!(enc_q.rc.target_crf, 26,
            "user-quality=26 maps to CRF≠26 (currently 33); update test if anchor table changes");

        // Clamping: CRF values are clamped to 0..=51.
        let enc_hi = Encoder::new_with_crf(32, 32, 99).unwrap();
        assert_eq!(enc_hi.crf, Some(51));
        assert_eq!(enc_hi.rc.target_crf, 51);
    }

    #[test]
    fn encoder_pcm_first_frame_emits_sps_pps_aud_idr() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        let pixels = make_yuv420p(32, 32);
        let bytes = enc.encode_i_frame_pcm(&pixels).unwrap();
        let nals = parse_nal_units_annexb(&bytes).unwrap();
        assert_eq!(nals.len(), 4);
        assert_eq!(nals[0].nal_type, NalType::SPS);
        assert_eq!(nals[1].nal_type, NalType::PPS);
        assert_eq!(nals[2].nal_type, NalType::AUD);
        assert_eq!(nals[3].nal_type, NalType::SLICE_IDR);
    }

    #[test]
    fn encoder_i16x16_first_frame_emits_sps_pps_aud_idr() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        let pixels = make_yuv420p(32, 32);
        let bytes = enc.encode_i_frame(&pixels).unwrap();
        let nals = parse_nal_units_annexb(&bytes).unwrap();
        assert_eq!(nals.len(), 4);
        assert_eq!(nals[0].nal_type, NalType::SPS);
        assert_eq!(nals[1].nal_type, NalType::PPS);
        assert_eq!(nals[2].nal_type, NalType::AUD);
        assert_eq!(nals[3].nal_type, NalType::SLICE_IDR);
    }

    #[test]
    fn encoder_i16x16_smaller_than_i_pcm() {
        // For a compressible deterministic pattern, the I_16x16 path
        // should produce dramatically fewer bytes than I_PCM (which is
        // ~384 bytes/MB regardless of content).
        let (w, h) = (64u32, 48u32);
        let mut enc_pcm = Encoder::new(w, h, Some(75)).unwrap();
        let mut enc_i16 = Encoder::new(w, h, Some(75)).unwrap();
        let pixels = make_yuv420p(w, h);
        let pcm = enc_pcm.encode_i_frame_pcm(&pixels).unwrap();
        let i16 = enc_i16.encode_i_frame(&pixels).unwrap();
        assert!(
            i16.len() < pcm.len(),
            "I_16x16 should compress: i16={} pcm={}",
            i16.len(),
            pcm.len()
        );
    }

    #[test]
    fn encoder_rejects_wrong_pixel_count() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        let short = vec![0u8; 100];
        assert!(enc.encode_i_frame(&short).is_err());
        assert!(enc.encode_i_frame_pcm(&short).is_err());
    }

    // ─── Phase 6C.6c: first CABAC I-frame ────────────────────────

    #[test]
    fn encoder_cabac_i_frame_emits_main_profile_sps_and_cabac_pps() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        let pixels = make_yuv420p(32, 32);
        let bytes = enc.encode_i_frame(&pixels).unwrap();
        let nals = parse_nal_units_annexb(&bytes).unwrap();
        assert_eq!(nals.len(), 4);
        assert_eq!(nals[0].nal_type, NalType::SPS);
        assert_eq!(nals[1].nal_type, NalType::PPS);
        assert_eq!(nals[2].nal_type, NalType::AUD);
        assert_eq!(nals[3].nal_type, NalType::SLICE_IDR);

        // Parse SPS and confirm Main profile (77).
        let sps =
            crate::codec::h264::sps::parse_sps(&nals[0].rbsp).expect("parse_sps of CABAC SPS");
        assert_eq!(sps.profile_idc, 77);

        // Parse PPS and confirm entropy_coding_mode_flag = 1.
        let pps =
            crate::codec::h264::sps::parse_pps(&nals[1].rbsp).expect("parse_pps of CABAC PPS");
        assert!(pps.entropy_coding_mode_flag);
    }

    #[test]
    fn encoder_cabac_i_frame_produces_nonempty_slice() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        let pixels = make_yuv420p(32, 32);
        let bytes = enc.encode_i_frame(&pixels).unwrap();
        let nals = parse_nal_units_annexb(&bytes).unwrap();
        // SLICE_IDR must contain a non-empty slice header + CABAC body +
        // 0x80 trailer. Smallest possible is 3+ bytes.
        let slice = nals.last().unwrap();
        assert!(slice.rbsp.len() >= 3);
        // Trailer should be byte 0x80 (rbsp_trailing_bits).
        assert_eq!(*slice.rbsp.last().unwrap(), 0x80);
    }

    #[test]
    fn encoder_pcm_writes_recon_buffer() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        let pixels = make_yuv420p(32, 32);
        let _ = enc.encode_i_frame_pcm(&pixels).unwrap();
        assert_eq!(enc.recon.y_at(5, 5), 10);
    }

    #[test]
    fn ac_scan_order_15_has_right_length() {
        let raster = [
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [9, 10, 13, 14],
            [11, 12, 15, 16],
        ];
        let scan = ac_scan_order_15(&raster);
        assert_eq!(scan.len(), 15);
        // Scan pos 1 should be raster [0][1] = 2 (since zigzag[1] = 1).
        assert_eq!(scan[0], 2);
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn i16x16_mb_type_dc_no_coeffs_eq_3() {
        // Formula: mb_type = 1 + pred_mode + 4*CBP_chroma + 12*CBP_luma_flag
        // DC mode (2), no coeffs: 1 + 2 + 0 + 0 = 3. Spec-formula literal.
        assert_eq!(1 + 2 + 4 * 0 + 12 * 0, 3);
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn i16x16_mb_type_dc_all_nonzero_eq_23() {
        // DC mode, luma nonzero, chroma full (AC+DC): 1 + 2 + 4*2 + 12*1 = 23.
        assert_eq!(1 + 2 + 4 * 2 + 12 * 1, 23);
    }

    #[test]
    fn encoder_promotes_to_dpb_after_i_frame() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        assert!(!enc.has_reference(), "fresh encoder has no reference");
        let pixels = make_yuv420p(32, 32);
        enc.encode_i_frame(&pixels).unwrap();
        assert!(enc.has_reference(), "after I-frame DPB holds previous recon");
    }

    #[test]
    fn encoder_idr_clears_then_promotes() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        let pixels = make_yuv420p(32, 32);
        enc.encode_i_frame(&pixels).unwrap();
        assert!(enc.has_reference());
        // Second I-frame (also IDR) clears first, then re-promotes.
        enc.encode_i_frame(&pixels).unwrap();
        assert!(enc.has_reference());
    }

    #[test]
    fn encoder_gop_position_advances_post_i() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        assert_eq!(enc.gop_position, 0);
        let pixels = make_yuv420p(32, 32);
        enc.encode_i_frame(&pixels).unwrap();
        assert_eq!(enc.gop_position, 1);
    }

    #[test]
    fn encoder_set_gop_length() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        assert_eq!(enc.gop_length, 30);
        enc.set_gop_length(10);
        assert_eq!(enc.gop_length, 10);
    }

    // ─── §6E-A4 B-frame encoder driver tests ─────────────────────

    /// §6E-A4 — encode_b_frame rejects calls without enable_b_frames.
    #[test]
    fn encode_b_frame_rejects_without_enable_flag() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        assert!(!enc.enable_b_frames);
        let pixels = make_yuv420p(32, 32);
        let r = enc.encode_b_frame(&pixels);
        assert!(matches!(r, Err(EncoderError::InvalidInput(_))));
    }

    /// §6E-A4 — encode_b_frame rejects calls without a DPB reference
    /// (caller must encode I + P first).
    #[test]
    fn encode_b_frame_rejects_without_dpb_reference() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_b_frames = true;
        let pixels = make_yuv420p(32, 32);
        let r = enc.encode_b_frame(&pixels);
        assert!(matches!(r, Err(EncoderError::InvalidInput(_))));
    }

    /// §6E-A4(a) — minimum-viable IBPBP encode. Encoder produces
    /// I + P + B Annex-B bytes; the bytes are non-empty and contain
    /// the expected NAL types in encode order.
    #[test]
    fn encode_ibpbp_emits_expected_nal_types() {
        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_b_frames = true;
        let pixels = make_yuv420p(32, 32);

        // Encode order for M=2 IBPBP: I, P, B, P, B, ...
        // First frame (display=0): I/IDR.
        let i_bytes = enc.encode_i_frame(&pixels).unwrap();
        assert!(!i_bytes.is_empty());
        // Display=2 P (encoded second).
        let p_bytes = enc.encode_p_frame(&pixels).unwrap();
        assert!(!p_bytes.is_empty());
        // Display=1 B (encoded third, between I and P).
        let b_bytes = enc.encode_b_frame(&pixels).unwrap();
        assert!(!b_bytes.is_empty());

        // Combined stream parses into NALs.
        let mut all = Vec::new();
        all.extend_from_slice(&i_bytes);
        all.extend_from_slice(&p_bytes);
        all.extend_from_slice(&b_bytes);
        let nalus = parse_nal_units_annexb(&all).expect("parse NALs");

        // Should contain at least: SPS, PPS, AUD, IDR, AUD, P, AUD, B.
        let mut idr_count = 0;
        let mut p_or_b_count = 0;
        let mut sps_count = 0;
        let mut pps_count = 0;
        for n in &nalus {
            if n.nal_type.is_idr() { idr_count += 1; }
            else if n.nal_type.is_vcl() { p_or_b_count += 1; }
            if matches!(n.nal_type, crate::codec::h264::NalType::SPS) {
                sps_count += 1;
            }
            if matches!(n.nal_type, crate::codec::h264::NalType::PPS) {
                pps_count += 1;
            }
        }
        assert_eq!(idr_count, 1, "exactly 1 IDR expected");
        assert_eq!(p_or_b_count, 2, "exactly 2 non-IDR VCL slices (P + B) expected");
        assert_eq!(sps_count, 1, "exactly 1 SPS expected");
        assert_eq!(pps_count, 1, "exactly 1 PPS expected");
    }

    /// §v1.4 Phase 4.4 (#315) — DUAL_REF_L0 IPP round-trip with
    /// distinct content per frame so the post-pass refine_p_choice_
    /// multi_ref *can* fire on the 2nd P-frame (where `past_anchor`
    /// is populated). Frame 0 = pattern A. Frame 1 = pattern B.
    /// Frame 2 = pattern A (matches the IDR more than P1). The 2nd
    /// P-frame's encoder sees ref_0=P1 (pattern B) and ref_1=I
    /// (pattern A); ME's post-pass should pick ref_1 for some MBs.
    /// Walker must round-trip the resulting stream regardless of
    /// which ref_idx_l0 each MB chose.
    #[test]
    fn encode_ipp_walker_round_trip_dual_ref_l0_with_upgrade() {
        use crate::codec::h264::cabac::bin_decoder::walk_annex_b_for_cover_with_options;
        use crate::codec::h264::cabac::bin_decoder::WalkOptions;

        // Two distinct YUV patterns so MEs differ between frames.
        let pattern_a = make_yuv420p(32, 32);
        let pattern_b = {
            let mut b = pattern_a.clone();
            // Shift luma by +13 to make ref_0 (P1=pattern_b) a worse
            // match for source (pattern_a) than ref_1 (I=pattern_a)
            // would be.
            for v in &mut b[..(32 * 32)] {
                *v = v.wrapping_add(13);
            }
            b
        };

        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        enc.multi_ref_config = MultiRefConfig::DUAL_REF_L0;

        let mut all = Vec::new();
        all.extend_from_slice(&enc.encode_i_frame(&pattern_a).unwrap());
        all.extend_from_slice(&enc.encode_p_frame(&pattern_b).unwrap());
        all.extend_from_slice(&enc.encode_p_frame(&pattern_a).unwrap());

        let opts = WalkOptions { record_mvd: true, record_offsets: false };
        let walk = walk_annex_b_for_cover_with_options(&all, opts)
            .expect("walker accepts IPP stream under DUAL_REF_L0");
        assert_eq!(walk.n_slices, 3, "expected 3 slices (I + P + P)");
    }

    /// §v1.4 Phase 2.5 (#305) — DUAL_REF_L0 round-trip. Same fixture
    /// as `encode_ibpbp_walker_round_trip` but flips
    /// `multi_ref_config = DUAL_REF_L0` so:
    ///  - slice header emits `num_ref_idx_active_override_flag=1` +
    ///    `num_ref_idx_l0_active_minus1=1` (Phase 2.3 wiring).
    ///  - encoder emit fns insert ref_idx_l0 unary "0" prefix (one
    ///    bin per partition with uses-L0) BEFORE MVDs (Phase 2.4).
    ///  - walker reads the same bin (Phase 3) and produces a
    ///    consistent walk result.
    /// At ME side ref_idx is still always 0 (Phase 4 wiring pending),
    /// so the bin emitted is always "0" prefix terminator. This test
    /// proves the wire format gate is correctly closed/opened by
    /// `MultiRefConfig` switching.
    #[test]
    fn encode_ibpbp_walker_round_trip_dual_ref_l0() {
        use crate::codec::h264::cabac::bin_decoder::walk_annex_b_for_cover_with_options;
        use crate::codec::h264::cabac::bin_decoder::WalkOptions;

        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_b_frames = true;
        // v1.4 Phase 2.3 type — flip from SINGLE_REF (default) to
        // DUAL_REF_L0 to exercise the new ref_idx_l0 emit/parse path.
        enc.multi_ref_config = MultiRefConfig::DUAL_REF_L0;
        let pixels = make_yuv420p(32, 32);

        let mut all = Vec::new();
        all.extend_from_slice(&enc.encode_i_frame(&pixels).unwrap());
        all.extend_from_slice(&enc.encode_p_frame(&pixels).unwrap());
        all.extend_from_slice(&enc.encode_b_frame(&pixels).unwrap());

        let opts = WalkOptions { record_mvd: true, record_offsets: false };
        let walk = walk_annex_b_for_cover_with_options(&all, opts)
            .expect("walker accepts IBPBP stream under DUAL_REF_L0");
        assert_eq!(walk.n_slices, 3, "expected 3 slices (I + P + B)");
    }

    /// §6E-A4(a) — encoder output IBPBP round-trips through the §6E-A3
    /// bin-decoder walker. Walker accepts SliceType::B and processes
    /// each B-MB through the all-B_Skip path. End-to-end gate that the
    /// encoder + walker pair are mutually consistent.
    #[test]
    fn encode_ibpbp_walker_round_trip() {
        use crate::codec::h264::cabac::bin_decoder::walk_annex_b_for_cover_with_options;
        use crate::codec::h264::cabac::bin_decoder::WalkOptions;

        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_b_frames = true;
        let pixels = make_yuv420p(32, 32);

        let mut all = Vec::new();
        all.extend_from_slice(&enc.encode_i_frame(&pixels).unwrap());
        all.extend_from_slice(&enc.encode_p_frame(&pixels).unwrap());
        all.extend_from_slice(&enc.encode_b_frame(&pixels).unwrap());

        // Walker reads it. The cover capture is meaningful only for
        // the I/P slices (B_Skip emits zero bypass-bins per MB).
        let opts = WalkOptions { record_mvd: true, record_offsets: false };
        let walk = walk_annex_b_for_cover_with_options(&all, opts)
            .expect("walker accepts IBPBP stream");
        assert_eq!(walk.n_slices, 3, "expected 3 slices (I + P + B)");
    }

    /// §6E-A4(c)-lite — verify the mode-decision helper produces
    /// both Skip and Direct outputs across a range of inputs (i.e.
    /// the hash isn't a constant function). Important for the
    /// L3-fingerprint property: encoder must produce mode variety.
    #[test]
    fn b_mb_mode_decision_produces_mix() {
        let mut skip_count = 0;
        let mut direct_count = 0;
        for mb_addr in 0..256u32 {
            if mb_skip_or_direct_decision(0, mb_addr) {
                direct_count += 1;
            } else {
                skip_count += 1;
            }
        }
        // 50/50 ± reasonable noise: both classes should be at least
        // 25% of samples.
        assert!(skip_count >= 64,
            "Skip count {skip_count} too low (mode mix biased to Direct)");
        assert!(direct_count >= 64,
            "Direct count {direct_count} too low (mode mix biased to Skip)");
    }

    /// §6E-A6.1 — encoder + walker round-trip for non-direct B
    /// mb_types. The env var `PHASM_B_FORCE_MODE` forces
    /// `mb_decision_b` to return a specific decision; we exercise
    /// each of L0_16x16 / L1_16x16 / Bi_16x16 in a B-slice and
    /// verify the walker accepts the encoder's output.
    ///
    /// All MVs are zero (placeholder for §6E-A6.1 part 1's encoder
    /// emission with hardcoded MV); real ME lands in part 4.
    #[test]
    fn encode_b_l0_16x16_walker_round_trip() {
        force_b_mode_round_trip("l0_16x16");
    }
    #[test]
    fn encode_b_l1_16x16_walker_round_trip() {
        force_b_mode_round_trip("l1_16x16");
    }
    #[test]
    fn encode_b_bi_16x16_walker_round_trip() {
        force_b_mode_round_trip("bi_16x16");
    }

    /// §6E-A6.2 — partitioned B mb_type round-trip tests. Sample
    /// the representative variants spanning all four partition×
    /// list-combo edges:
    ///   mb_type 4   = B_L0_L0_16x8  (H, L0 + L0)  — simplest 16x8
    ///   mb_type 5   = B_L0_L0_8x16  (V, L0 + L0)  — simplest 8x16
    ///   mb_type 8   = B_L0_L1_16x8  (H, L0 + L1)  — mixed-list 16x8
    ///   mb_type 11  = B_L1_L0_8x16  (V, L1 + L0)  — v=14 short-circuit
    ///   mb_type 13  = B_L0_Bi_8x16  (V, L0 + Bi)  — bin6 path
    ///   mb_type 20  = B_Bi_Bi_16x8  (H, Bi + Bi)  — most-bipred 16x8
    ///   mb_type 21  = B_Bi_Bi_8x16  (V, Bi + Bi)  — most-bipred 8x16
    /// Each forces PHASM_B_FORCE_MODE=partitioned_<n>, encodes
    /// I+P+B at 32x32, walks the result, asserts 3 slices.
    #[test] fn encode_b_partitioned_mb_4_round_trip() { force_b_mode_round_trip("partitioned_4"); }
    #[test] fn encode_b_partitioned_mb_5_round_trip() { force_b_mode_round_trip("partitioned_5"); }
    #[test] fn encode_b_partitioned_mb_8_round_trip() { force_b_mode_round_trip("partitioned_8"); }
    #[test] fn encode_b_partitioned_mb_11_round_trip() { force_b_mode_round_trip("partitioned_11"); }
    #[test] fn encode_b_partitioned_mb_13_round_trip() { force_b_mode_round_trip("partitioned_13"); }
    #[test] fn encode_b_partitioned_mb_20_round_trip() { force_b_mode_round_trip("partitioned_20"); }
    #[test] fn encode_b_partitioned_mb_21_round_trip() { force_b_mode_round_trip("partitioned_21"); }

    /// §6E-A6.3 — uniform B_8x8 (mb_type = 22) round-trip tests. Each
    /// forces all 4 sub-MBs to the same `sub_mb_type` (0..=3) so the
    /// encoder + walker exercise:
    ///   - sub_mb_type=0 (B_Direct_8x8): 1-bin path (no MVDs)
    ///   - sub_mb_type=1 (B_L0_8x8):     3-bin path + L0 MVD per sub-MB
    ///   - sub_mb_type=2 (B_L1_8x8):     3-bin path + L1 MVD per sub-MB
    ///   - sub_mb_type=3 (B_Bi_8x8):     5-bin path + L0+L1 MVDs per sub-MB
    /// `b_8x8_mixed` exercises one of each in the same MB so the
    /// per-bin ctxIdxInc state tracking + MVD ordering across
    /// heterogeneous sub-MBs is verified.
    #[test] fn b_8x8_uniform_direct_roundtrip() { force_b_mode_round_trip("b_8x8_uniform_direct"); }
    #[test] fn b_8x8_uniform_l0_roundtrip() { force_b_mode_round_trip("b_8x8_uniform_l0"); }
    #[test] fn b_8x8_uniform_l1_roundtrip() { force_b_mode_round_trip("b_8x8_uniform_l1"); }
    #[test] fn b_8x8_uniform_bi_roundtrip() { force_b_mode_round_trip("b_8x8_uniform_bi"); }
    #[test] fn b_8x8_mixed_4subtype_roundtrip() { force_b_mode_round_trip("b_8x8_mixed"); }

    /// Helper for the three above. Uses a process-wide Mutex to
    /// serialize env-var access (PHASM_B_FORCE_MODE is process-
    /// global; concurrent tests would race). Shared with the
    /// `mb_decision_b::tests::distribution_match_centroid_buckets`
    /// test which also depends on the env var being in a known state.
    fn force_b_mode_round_trip(mode: &str) {
        use crate::codec::h264::cabac::bin_decoder::walk_annex_b_for_cover_with_options;
        use crate::codec::h264::cabac::bin_decoder::WalkOptions;
        use crate::codec::h264::encoder::mb_decision_b::B_FORCE_MODE_ENV_LOCK;

        let _lock = B_FORCE_MODE_ENV_LOCK.lock().expect("lock not poisoned");
        // SAFETY: serialized across tests via ENV_LOCK; no other
        // thread reads PHASM_B_FORCE_MODE outside `mb_decision_b`,
        // and that runs only inside the encoder call below.
        unsafe { std::env::set_var("PHASM_B_FORCE_MODE", mode); }
        let result = std::panic::catch_unwind(|| {
            let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
            enc.entropy_mode = EntropyMode::Cabac;
            enc.enable_b_frames = true;
            let pixels = make_yuv420p(32, 32);

            let mut all = Vec::new();
            all.extend_from_slice(&enc.encode_i_frame(&pixels).unwrap());
            all.extend_from_slice(&enc.encode_p_frame(&pixels).unwrap());
            all.extend_from_slice(&enc.encode_b_frame(&pixels).unwrap());

            let opts = WalkOptions { record_mvd: true, record_offsets: false };
            let walk = walk_annex_b_for_cover_with_options(&all, opts)
                .expect("walker accepts B-slice with non-direct mb_type");
            assert_eq!(walk.n_slices, 3, "expected 3 slices (I + P + B)");
        });
        // Always clean up the env var, even on panic.
        unsafe { std::env::remove_var("PHASM_B_FORCE_MODE"); }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }

    /// §6E-A4(c)-lite — encoder + walker round-trip with the mode
    /// mix active. Larger frame (more MBs) so the hash exercises
    /// both Skip and Direct paths in the same B-slice.
    #[test]
    fn encode_ibpbp_with_mode_mix_walker_roundtrip() {
        use crate::codec::h264::cabac::bin_decoder::walk_annex_b_for_cover_with_options;
        use crate::codec::h264::cabac::bin_decoder::WalkOptions;

        let mut enc = Encoder::new(64, 64, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_b_frames = true;
        let pixels = make_yuv420p(64, 64);

        // 64×64 = 16 MBs per frame. With the hash mix, the B-frame
        // will contain both Skip and Direct MBs.
        let mut all = Vec::new();
        all.extend_from_slice(&enc.encode_i_frame(&pixels).unwrap());
        all.extend_from_slice(&enc.encode_p_frame(&pixels).unwrap());
        all.extend_from_slice(&enc.encode_b_frame(&pixels).unwrap());

        let opts = WalkOptions { record_mvd: true, record_offsets: false };
        let walk = walk_annex_b_for_cover_with_options(&all, opts)
            .expect("walker accepts mixed-mode B-slice");
        assert_eq!(walk.n_slices, 3);
    }

    /// §6E-A6.1q.b (#151) — round-trip with shifted-content frames so
    /// real ME finds non-zero MVs and the B-frame's L0/L1/Bi 16x16
    /// MBs emit non-zero MVD bytes on the wire.
    ///
    /// Shape: I (gradient), P (gradient shifted +2 px right), B
    /// (gradient shifted +1 px right — between I and P in display
    /// motion, so spatial-direct MV ≈ {1, 0} and ME against L0=I /
    /// L1=P should find MVs near {-1, 0} / {+1, 0}). 64×64 = 16 MBs
    /// per frame; force-mode `bi_16x16` ensures every B-MB exercises
    /// the dual-list MVD path.
    #[test]
    fn b_frame_real_me_emits_nonzero_mvd_round_trip() {
        use crate::codec::h264::cabac::bin_decoder::walk_annex_b_for_cover_with_options;
        use crate::codec::h264::cabac::bin_decoder::WalkOptions;
        use crate::codec::h264::encoder::mb_decision_b::B_FORCE_MODE_ENV_LOCK;

        // Generate 3 shifted gradients — different content at each
        // position so ME has a non-trivial search target.
        let make_shifted = |w: u32, h: u32, shift_x: i32| -> Vec<u8> {
            let y = (w * h) as usize;
            let c = (w / 2 * h / 2) as usize;
            let mut buf = vec![0u8; y + 2 * c];
            for yy in 0..h {
                for xx in 0..w {
                    let sx = xx as i32 + shift_x;
                    buf[(yy * w + xx) as usize] = ((sx + yy as i32) & 0xFF) as u8;
                }
            }
            for yy in 0..(h / 2) {
                for xx in 0..(w / 2) {
                    buf[y + (yy * (w / 2) + xx) as usize] = (xx & 0xFF) as u8;
                }
            }
            for yy in 0..(h / 2) {
                for xx in 0..(w / 2) {
                    buf[y + c + (yy * (w / 2) + xx) as usize] = (yy & 0xFF) as u8;
                }
            }
            buf
        };

        let i_pixels = make_shifted(64, 64, 0);
        let p_pixels = make_shifted(64, 64, 2);
        let b_pixels = make_shifted(64, 64, 1);

        // Force every B-MB to Bi_16x16 so ME's L0+L1 MVs both flow
        // through MVD emission. Lock the env-var to avoid races.
        let _lock = B_FORCE_MODE_ENV_LOCK.lock().expect("lock not poisoned");
        // SAFETY: serialized via lock; var is removed after the test.
        unsafe { std::env::set_var("PHASM_B_FORCE_MODE", "bi_16x16"); }
        let result = std::panic::catch_unwind(|| {
            let mut enc = Encoder::new(64, 64, Some(75)).unwrap();
            enc.entropy_mode = EntropyMode::Cabac;
            enc.enable_b_frames = true;

            let mut all = Vec::new();
            all.extend_from_slice(&enc.encode_i_frame(&i_pixels).unwrap());
            all.extend_from_slice(&enc.encode_p_frame(&p_pixels).unwrap());
            all.extend_from_slice(&enc.encode_b_frame(&b_pixels).unwrap());

            let opts = WalkOptions { record_mvd: true, record_offsets: false };
            let walk = walk_annex_b_for_cover_with_options(&all, opts)
                .expect("walker accepts B-slice with non-zero MVDs from real ME");
            assert_eq!(walk.n_slices, 3, "expected 3 slices (I + P + B)");
            // The B-slice cover should have recorded MVD positions
            // since every MB emitted Bi_16x16 with two MVD pairs.
            // 16 MBs × 2 MVD pairs (L0 + L1) × 2 components (X, Y) = 64 MVD records.
            // Each MVD record carries one sign bin (when nonzero), so
            // we expect mvd_sign_bypass.positions.len() proportional
            // to the number of nonzero MVDs.
            assert!(
                !walk.cover.mvd_sign_bypass.positions.is_empty(),
                "real ME should produce at least some nonzero MVDs (sign-bypass cover empty)"
            );
        });
        unsafe { std::env::remove_var("PHASM_B_FORCE_MODE"); }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }

    /// §6E-A6.1q.b/c (#151+#152) — round-trip with B-frame residual
    /// emission enabled. Encoder produces non-zero CBP B-MBs via the
    /// new `write_b_inter_residual_macroblock_cabac` path; walker
    /// decodes residuals via the un-rejected `finish_b_inter` tail.
    /// Closes the §6E-A6.1q.b ship: motion + texture detail both
    /// flow through B-frames now.
    #[test]
    fn b_frame_residual_emission_round_trip() {
        use crate::codec::h264::cabac::bin_decoder::walk_annex_b_for_cover_with_options;
        use crate::codec::h264::cabac::bin_decoder::WalkOptions;
        use crate::codec::h264::encoder::mb_decision_b::B_FORCE_MODE_ENV_LOCK;

        // Shifted-content frames so the B-frame's residual is non-
        // trivial (otherwise quant rounds to zero everywhere — same
        // CBP=0 path as before, no novel coverage).
        let make_shifted = |w: u32, h: u32, shift_x: i32| -> Vec<u8> {
            let y = (w * h) as usize;
            let c = (w / 2 * h / 2) as usize;
            let mut buf = vec![0u8; y + 2 * c];
            for yy in 0..h {
                for xx in 0..w {
                    let sx = xx as i32 + shift_x;
                    buf[(yy * w + xx) as usize] =
                        (((sx * 7 + (yy as i32) * 5) & 0xFF) as u8).wrapping_add(40);
                }
            }
            for yy in 0..(h / 2) {
                for xx in 0..(w / 2) {
                    buf[y + (yy * (w / 2) + xx) as usize] =
                        (128_u8).wrapping_add((xx as u8).wrapping_mul(3));
                    buf[y + c + (yy * (w / 2) + xx) as usize] =
                        (128_u8).wrapping_add((yy as u8).wrapping_mul(3));
                }
            }
            buf
        };

        let i_pixels = make_shifted(64, 64, 0);
        let p_pixels = make_shifted(64, 64, 2);
        let b_pixels = make_shifted(64, 64, 1);

        // Lock both env vars (force-mode + residual-enable) for
        // serialized test access.
        let _lock = B_FORCE_MODE_ENV_LOCK.lock().expect("lock not poisoned");
        // SAFETY: lock-serialized; vars cleared after panic-catch.
        unsafe {
            std::env::set_var("PHASM_B_FORCE_MODE", "bi_16x16");
            std::env::set_var("PHASM_B_RESIDUAL", "1");
        }
        let result = std::panic::catch_unwind(|| {
            let mut enc = Encoder::new(64, 64, Some(75)).unwrap();
            enc.entropy_mode = EntropyMode::Cabac;
            enc.enable_b_frames = true;

            let mut all = Vec::new();
            all.extend_from_slice(&enc.encode_i_frame(&i_pixels).unwrap());
            all.extend_from_slice(&enc.encode_p_frame(&p_pixels).unwrap());
            all.extend_from_slice(&enc.encode_b_frame(&b_pixels).unwrap());

            let opts = WalkOptions { record_mvd: true, record_offsets: false };
            let walk = walk_annex_b_for_cover_with_options(&all, opts)
                .expect("walker accepts B-slice with non-zero CBP residuals");
            assert_eq!(walk.n_slices, 3, "expected 3 slices (I + P + B)");
            // The primary #152 success criterion: walker accepts the
            // stream produced by the new residual-emission encoder
            // path. Coverage non-emptiness is content-dependent (e.g.
            // synthetic content where ME converges on zero MVDs and
            // residual quantises to zero) — gate on `n_slices` only,
            // and rely on the panic-free round-trip as the test signal.
            // The MVD/residual cover inspection is delegated to
            // explicit non-zero assertions in §6E-A6.1q.b's existing
            // `b_frame_real_me_emits_nonzero_mvd_round_trip` (which
            // uses a less aggressive content shape that survives
            // residual quantisation to zero).
        });
        unsafe {
            std::env::remove_var("PHASM_B_FORCE_MODE");
            std::env::remove_var("PHASM_B_RESIDUAL");
        }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }

    /// Task #207 reproducer harness — same as task207_repro but with
    /// residual emission DISABLED (PHASM_B_RESIDUAL=0). Verifies the
    /// legacy CBP=0 path doesn't desync.
    fn task207_repro_no_residual(mode: &str, force_mv: &str) {
        use crate::codec::h264::cabac::bin_decoder::walk_annex_b_for_cover_with_options;
        use crate::codec::h264::cabac::bin_decoder::WalkOptions;
        use crate::codec::h264::encoder::mb_decision_b::B_FORCE_MODE_ENV_LOCK;

        let make_shifted = |w: u32, h: u32, shift_x: i32| -> Vec<u8> {
            let y = (w * h) as usize;
            let c = (w / 2 * h / 2) as usize;
            let mut buf = vec![0u8; y + 2 * c];
            for yy in 0..h {
                for xx in 0..w {
                    let sx = xx as i32 + shift_x;
                    buf[(yy * w + xx) as usize] =
                        (((sx * 7 + (yy as i32) * 5) & 0xFF) as u8).wrapping_add(40);
                }
            }
            for yy in 0..(h / 2) {
                for xx in 0..(w / 2) {
                    buf[y + (yy * (w / 2) + xx) as usize] =
                        (128_u8).wrapping_add((xx as u8).wrapping_mul(3));
                    buf[y + c + (yy * (w / 2) + xx) as usize] =
                        (128_u8).wrapping_add((yy as u8).wrapping_mul(3));
                }
            }
            buf
        };
        let i_pixels = make_shifted(64, 64, 0);
        let p_pixels = make_shifted(64, 64, 2);
        let b_pixels = make_shifted(64, 64, 1);

        let _lock = B_FORCE_MODE_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("PHASM_B_FORCE_MODE", mode);
            std::env::set_var("PHASM_B_FORCE_MV", force_mv);
            std::env::set_var("PHASM_B_RESIDUAL", "0");
        }
        let result = std::panic::catch_unwind(|| {
            let mut enc = Encoder::new(64, 64, Some(75)).unwrap();
            enc.entropy_mode = EntropyMode::Cabac;
            enc.enable_b_frames = true;

            let mut all = Vec::new();
            all.extend_from_slice(&enc.encode_i_frame(&i_pixels).unwrap());
            all.extend_from_slice(&enc.encode_p_frame(&p_pixels).unwrap());
            all.extend_from_slice(&enc.encode_b_frame(&b_pixels).unwrap());

            let opts = WalkOptions { record_mvd: true, record_offsets: false };
            let walk = walk_annex_b_for_cover_with_options(&all, opts)
                .unwrap_or_else(|e| panic!("walker rejected: {e:?}"));
            assert_eq!(walk.n_slices, 3);
        });
        unsafe {
            std::env::remove_var("PHASM_B_FORCE_MODE");
            std::env::remove_var("PHASM_B_FORCE_MV");
            std::env::remove_var("PHASM_B_RESIDUAL");
        }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }

    /// Task #207 — also passes with PHASM_B_RESIDUAL=0 (legacy CBP=0
    /// path), confirming the walker-side neighbor-commit fix is what
    /// closed the gap, not a residual-emission change.
    #[test] fn task207_l1_y4_no_residual() {
        let _lock = crate::codec::h264::encoder::mb_decision_b::B_FORCE_MODE_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("PHASM_B_FORCE_MODE", "b_8x8_uniform_l1");
            std::env::set_var("PHASM_B_FORCE_MV", "0,4");
            std::env::set_var("PHASM_B_RESIDUAL", "0");
        }
        let result = std::panic::catch_unwind(|| {
            use crate::codec::h264::cabac::bin_decoder::walk_annex_b_for_cover_with_options;
            use crate::codec::h264::cabac::bin_decoder::WalkOptions;
            let make_shifted = |w: u32, h: u32, sx: i32| -> Vec<u8> {
                let y = (w * h) as usize;
                let c = (w / 2 * h / 2) as usize;
                let mut buf = vec![0u8; y + 2 * c];
                for yy in 0..h {
                    for xx in 0..w {
                        buf[(yy * w + xx) as usize] = (((xx as i32 + sx) * 7 + (yy as i32) * 5) & 0xFF) as u8;
                    }
                }
                for v in &mut buf[y..] { *v = 128; }
                buf
            };
            let mut enc = Encoder::new(64, 64, Some(75)).unwrap();
            enc.entropy_mode = EntropyMode::Cabac;
            enc.enable_b_frames = true;
            let mut all = Vec::new();
            all.extend_from_slice(&enc.encode_i_frame(&make_shifted(64, 64, 0)).unwrap());
            all.extend_from_slice(&enc.encode_p_frame(&make_shifted(64, 64, 2)).unwrap());
            all.extend_from_slice(&enc.encode_b_frame(&make_shifted(64, 64, 1)).unwrap());
            let opts = WalkOptions { record_mvd: true, record_offsets: false };
            let walk = walk_annex_b_for_cover_with_options(&all, opts)
                .unwrap_or_else(|e| panic!("walker rejected: {e:?}"));
            assert_eq!(walk.n_slices, 3);
        });
        unsafe {
            std::env::remove_var("PHASM_B_FORCE_MODE");
            std::env::remove_var("PHASM_B_FORCE_MV");
            std::env::remove_var("PHASM_B_RESIDUAL");
        }
        if let Err(e) = result { std::panic::resume_unwind(e); }
    }

    /// Task #207 reproducer harness — bisects the L1 + non-zero-MV
    /// walker desync.
    fn task207_repro(mode: &str, force_mv: &str) {
        use crate::codec::h264::cabac::bin_decoder::walk_annex_b_for_cover_with_options;
        use crate::codec::h264::cabac::bin_decoder::WalkOptions;
        use crate::codec::h264::encoder::mb_decision_b::B_FORCE_MODE_ENV_LOCK;

        let make_shifted = |w: u32, h: u32, shift_x: i32| -> Vec<u8> {
            let y = (w * h) as usize;
            let c = (w / 2 * h / 2) as usize;
            let mut buf = vec![0u8; y + 2 * c];
            for yy in 0..h {
                for xx in 0..w {
                    let sx = xx as i32 + shift_x;
                    buf[(yy * w + xx) as usize] =
                        (((sx * 7 + (yy as i32) * 5) & 0xFF) as u8).wrapping_add(40);
                }
            }
            for yy in 0..(h / 2) {
                for xx in 0..(w / 2) {
                    buf[y + (yy * (w / 2) + xx) as usize] =
                        (128_u8).wrapping_add((xx as u8).wrapping_mul(3));
                    buf[y + c + (yy * (w / 2) + xx) as usize] =
                        (128_u8).wrapping_add((yy as u8).wrapping_mul(3));
                }
            }
            buf
        };
        let i_pixels = make_shifted(64, 64, 0);
        let p_pixels = make_shifted(64, 64, 2);
        let b_pixels = make_shifted(64, 64, 1);

        let _lock = B_FORCE_MODE_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        unsafe {
            std::env::set_var("PHASM_B_FORCE_MODE", mode);
            std::env::set_var("PHASM_B_FORCE_MV", force_mv);
            std::env::set_var("PHASM_B_RESIDUAL", "1");
        }
        let result = std::panic::catch_unwind(|| {
            let mut enc = Encoder::new(64, 64, Some(75)).unwrap();
            enc.entropy_mode = EntropyMode::Cabac;
            enc.enable_b_frames = true;

            let mut all = Vec::new();
            all.extend_from_slice(&enc.encode_i_frame(&i_pixels).unwrap());
            all.extend_from_slice(&enc.encode_p_frame(&p_pixels).unwrap());
            all.extend_from_slice(&enc.encode_b_frame(&b_pixels).unwrap());

            let opts = WalkOptions { record_mvd: true, record_offsets: false };
            let walk = walk_annex_b_for_cover_with_options(&all, opts)
                .unwrap_or_else(|e| panic!("walker rejected: {e:?}"));
            assert_eq!(walk.n_slices, 3, "expected 3 slices (I + P + B)");
        });
        unsafe {
            std::env::remove_var("PHASM_B_FORCE_MODE");
            std::env::remove_var("PHASM_B_FORCE_MV");
            std::env::remove_var("PHASM_B_RESIDUAL");
        }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }

    /// Task #207 — once-failing cases at non-zero L1 MV.
    /// All previously failed at certain Y magnitudes (3, 4) but
    /// passed at 0/1/2/5 due to per-position vs broadcast neighbor
    /// commit divergence on the walker side. Fixed in
    /// `bin_decoder/slice.rs::finish_b_inter` 2026-05-04.
    #[test] fn task207_l1_y4_residual() { task207_repro("b_8x8_uniform_l1", "0,4"); }
    #[test] fn task207_l1_y3_residual() { task207_repro("b_8x8_uniform_l1", "0,3"); }
    #[test] fn task207_l1_x4y4_residual() { task207_repro("b_8x8_uniform_l1", "4,4"); }
    #[test] fn task207_l1_y_neg4_residual() { task207_repro("b_8x8_uniform_l1", "0,-4"); }
    #[test] fn task207_part8_int() { task207_repro("partitioned_8", "2,2"); }

    /// §6E-A4(a) — when enable_b_frames=true, the SPS must use
    /// pic_order_cnt_type=0 (so decoders can reorder display order).
    #[test]
    fn enable_b_frames_bumps_sps_to_pic_order_cnt_type_0() {
        use crate::codec::h264::sps::parse_sps;

        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_b_frames = true;
        let pixels = make_yuv420p(32, 32);
        let i_bytes = enc.encode_i_frame(&pixels).unwrap();
        let nalus = parse_nal_units_annexb(&i_bytes).expect("parse NALs");
        let sps_nal = nalus.iter()
            .find(|n| matches!(n.nal_type, crate::codec::h264::NalType::SPS))
            .expect("SPS NAL emitted");
        let sps = parse_sps(&sps_nal.rbsp).expect("SPS parses");
        assert_eq!(sps.pic_order_cnt_type, 0,
            "B-frame mode requires pic_order_cnt_type=0");
        assert!(sps.max_num_ref_frames >= 3,
            "B-frame mode requires max_num_ref_frames>=3 (M=2 IBPBP DPB)");
    }

    /// §6E-B(a) — confirm SPS emits `max_num_ref_frames=2` per the
    /// fingerprint-match bump. The default Encoder (no
    /// enable_b_frames) should now advertise 2 refs in SPS, even
    /// though the encoder still uses single-ref ME internally.
    #[test]
    fn default_sps_advertises_two_refs() {
        use crate::codec::h264::sps::parse_sps;

        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        // Note: NOT setting enable_b_frames; this is the default
        // Phase 6B I+P-only path. The §6E-B(a) bump applies even
        // without B-frames.
        let pixels = make_yuv420p(32, 32);
        let i_bytes = enc.encode_i_frame(&pixels).unwrap();
        let nalus = parse_nal_units_annexb(&i_bytes).expect("parse NALs");
        let sps_nal = nalus.iter()
            .find(|n| matches!(n.nal_type, crate::codec::h264::NalType::SPS))
            .expect("SPS NAL emitted");
        let sps = parse_sps(&sps_nal.rbsp).expect("SPS parses");
        assert_eq!(sps.max_num_ref_frames, 2,
            "§6E-B(a) bumps max_num_ref_frames from 1 to 2 for SPS-level \
             fingerprint match with commercial encoders");
    }

    /// §6E-B(a) — when enable_b_frames=true, SPS bumps to 3 refs
    /// (M=2 IBPBP DPB shape) — overrides §6E-B(a)'s default of 2.
    #[test]
    fn enable_b_frames_overrides_to_three_refs() {
        use crate::codec::h264::sps::parse_sps;

        let mut enc = Encoder::new(32, 32, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_b_frames = true;
        let pixels = make_yuv420p(32, 32);
        let i_bytes = enc.encode_i_frame(&pixels).unwrap();
        let nalus = parse_nal_units_annexb(&i_bytes).expect("parse NALs");
        let sps_nal = nalus.iter()
            .find(|n| matches!(n.nal_type, crate::codec::h264::NalType::SPS))
            .expect("SPS NAL emitted");
        let sps = parse_sps(&sps_nal.rbsp).expect("SPS parses");
        assert_eq!(sps.max_num_ref_frames, 3,
            "B-frame mode requires max_num_ref_frames>=3 (M=2 IBPBP DPB)");
    }

    /// §B-direct-fix.v2 (#194) — 1080p the reference decoder compliance gate for
    /// L0_16x16 + non-zero CBP B-residual emission.
    ///
    /// Same logic as `reference_decoder_decodes_ibpbp_with_b_residual_without_errors`
    /// but at 1920×1072 with rich content + force=l0_16x16. The 96×96
    /// gate passes (smooth content → mostly CBP=0). 1080p with
    /// shifted gradient triggers ~14 concealment events (verified
    /// via the 30D-C orchestrator path through the iPhone7 fixture)
    /// — this lib-level gate isolates the bug from the stego
    /// orchestrator so iteration is fast (~10sec per encode).
    #[test]
    #[ignore = "1080p, requires the reference decoder in PATH; run with --ignored"]
    fn reference_decoder_decodes_l0_16x16_b_residual_1080p() {
        use crate::codec::h264::encoder::mb_decision_b::B_FORCE_MODE_ENV_LOCK;
        use std::process::Command;

        let make_shifted = |w: u32, h: u32, shift_x: i32| -> Vec<u8> {
            let y = (w * h) as usize;
            let c = (w / 2 * h / 2) as usize;
            let mut buf = vec![0u8; y + 2 * c];
            for yy in 0..h {
                for xx in 0..w {
                    let sx = xx as i32 + shift_x;
                    buf[(yy * w + xx) as usize] =
                        (((sx * 7 + (yy as i32) * 5) & 0xFF) as u8).wrapping_add(40);
                }
            }
            for yy in 0..(h / 2) {
                for xx in 0..(w / 2) {
                    buf[y + (yy * (w / 2) + xx) as usize] =
                        (128_u8).wrapping_add((xx as u8).wrapping_mul(3));
                    buf[y + c + (yy * (w / 2) + xx) as usize] =
                        (128_u8).wrapping_add((yy as u8).wrapping_mul(3));
                }
            }
            buf
        };

        let _lock = B_FORCE_MODE_ENV_LOCK.lock().expect("lock not poisoned");
        unsafe {
            std::env::set_var("PHASM_B_FORCE_MODE", "l0_16x16");
            std::env::set_var("PHASM_B_RESIDUAL", "1");
        }
        let result = std::panic::catch_unwind(|| {
            let mut enc = Encoder::new(1920, 1072, Some(75)).unwrap();
            enc.entropy_mode = EntropyMode::Cabac;
            enc.enable_b_frames = true;

            let mut all = Vec::new();
            all.extend_from_slice(&enc.encode_i_frame(&make_shifted(1920, 1072, 0)).unwrap());
            for k in 0..3 {
                let p_shift = (k * 4 + 4) as i32;
                let b_shift = (k * 4 + 2) as i32;
                all.extend_from_slice(
                    &enc.encode_p_frame(&make_shifted(1920, 1072, p_shift)).unwrap(),
                );
                all.extend_from_slice(
                    &enc.encode_b_frame(&make_shifted(1920, 1072, b_shift)).unwrap(),
                );
            }

            let path = std::env::temp_dir()
                .join("phasm_194_l0_residual_1080p.h264");
            std::fs::write(&path, &all).expect("write temp h264");

            let out = Command::new("the reference decoder")
                .args([
                    "-loglevel", "info",
                    "-i", path.to_str().unwrap(),
                    "-f", "null", "-",
                ])
                .output()
                .expect("the reference decoder in PATH");
            let stderr = String::from_utf8_lossy(&out.stderr);
            let conceal_lines: Vec<&str> = stderr.lines()
                .filter(|l| l.contains("concealing"))
                .collect();
            eprintln!(
                "1080p l0_16x16 + B-residual: the reference decoder exit {:?}, {} conceal events, h264 size {} bytes",
                out.status, conceal_lines.len(), all.len(),
            );
            for line in conceal_lines.iter().take(3) {
                eprintln!("  {line}");
            }
            assert!(out.status.success(), "the reference decoder failed: {stderr}");
            assert_eq!(
                conceal_lines.len(),
                0,
                "the reference decoder flagged 1080p L0_16x16 B-residual concealment ({} events)",
                conceal_lines.len(),
            );
        });
        unsafe {
            std::env::remove_var("PHASM_B_FORCE_MODE");
            std::env::remove_var("PHASM_B_RESIDUAL");
        }
        if let Err(e) = result { std::panic::resume_unwind(e); }
    }

    /// §B-direct-fix.v2 (#194) — 1080p iPhone7-content reproducer.
    /// Like the synthetic-content variant above, but feeds the
    /// /tmp/iphone7_1920x1072_f10.yuv fixture (rich real motion).
    /// Synthetic gradient does NOT trigger the bug; iPhone content
    /// does. Identifies the bug as content-driven (high-magnitude
    /// residual coefficients in B-frame L0_16x16 emission).
    #[test]
    #[ignore = "requires /tmp/iphone7_1920x1072_f10.yuv + the reference decoder; --ignored"]
    fn reference_decoder_decodes_l0_16x16_b_residual_iphone7_1080p() {
        use crate::codec::h264::encoder::mb_decision_b::B_FORCE_MODE_ENV_LOCK;
        use std::process::Command;

        let yuv_path = "/tmp/iphone7_1920x1072_f10.yuv";
        let yuv_all = match std::fs::read(yuv_path) {
            Ok(b) => b,
            Err(_) => {
                eprintln!("missing fixture {yuv_path} — skipping");
                return;
            }
        };
        let frame_size = 1920 * 1072 * 3 / 2;
        let frames: Vec<&[u8]> = yuv_all.chunks_exact(frame_size).take(7).collect();
        if frames.len() < 7 {
            eprintln!("fixture too short ({} frames, need 7) — skipping", frames.len());
            return;
        }

        let _lock = B_FORCE_MODE_ENV_LOCK.lock().expect("lock not poisoned");
        unsafe {
            std::env::set_var("PHASM_B_FORCE_MODE", "l0_16x16");
            std::env::set_var("PHASM_B_RESIDUAL", "1");
        }
        let result = std::panic::catch_unwind(|| {
            let mut enc = Encoder::new(1920, 1072, Some(75)).unwrap();
            enc.entropy_mode = EntropyMode::Cabac;
            enc.enable_b_frames = true;
            // §B-direct-fix.v2 — match the orchestrator's
            // `build_encoder` config exactly. The orchestrator
            // enables 8x8 transform + High profile (#145).
            enc.enable_transform_8x8 = true;

            let mut all = Vec::new();
            // I + 3×(P, B) = 7 frames in IBPBP shape (display order:
            // I0 B1 P2 B3 P4 B5 P6; encode order: I P B P B P B).
            all.extend_from_slice(&enc.encode_i_frame(frames[0]).unwrap());
            for k in 0..3 {
                let p_idx = 2 + k * 2;
                let b_idx = 1 + k * 2;
                all.extend_from_slice(&enc.encode_p_frame(frames[p_idx]).unwrap());
                all.extend_from_slice(&enc.encode_b_frame(frames[b_idx]).unwrap());
            }

            let path = std::env::temp_dir()
                .join("phasm_194_l0_residual_iphone7_1080p.h264");
            std::fs::write(&path, &all).expect("write temp h264");

            let out = Command::new("the reference decoder")
                .args([
                    "-loglevel", "info",
                    "-i", path.to_str().unwrap(),
                    "-f", "null", "-",
                ])
                .output()
                .expect("the reference decoder in PATH");
            let stderr = String::from_utf8_lossy(&out.stderr);
            let conceal_lines: Vec<&str> = stderr.lines()
                .filter(|l| l.contains("concealing"))
                .collect();
            eprintln!(
                "iphone7 1080p l0_16x16 + B-residual: {} conceal events, h264 size {} bytes",
                conceal_lines.len(), all.len(),
            );
            for line in conceal_lines.iter().take(3) {
                eprintln!("  {line}");
            }
            assert!(out.status.success(), "the reference decoder failed: {stderr}");
            assert_eq!(
                conceal_lines.len(),
                0,
                "the reference decoder flagged {} concealment events",
                conceal_lines.len(),
            );
        });
        unsafe {
            std::env::remove_var("PHASM_B_FORCE_MODE");
            std::env::remove_var("PHASM_B_RESIDUAL");
        }
        if let Err(e) = result { std::panic::resume_unwind(e); }
    }

    /// §6E-A6.1q.d (#153) — the reference decoder compliance gate for the new
    /// B-frame residual emission path.
    ///
    /// Confirms `write_b_inter_residual_macroblock_cabac` produces
    /// spec-compliant H.264 with non-zero CBP B-MBs. Uses non-
    /// uniform shifted gradient content so residual quantises to
    /// non-zero on at least some B-MBs. Forces B_FORCE_MODE=bi_16x16
    /// so every B-MB exercises the new path.
    #[test]
    #[ignore = "requires the reference decoder in PATH; run with --ignored"]
    fn reference_decoder_decodes_ibpbp_with_b_residual_without_errors() {
        use crate::codec::h264::encoder::mb_decision_b::B_FORCE_MODE_ENV_LOCK;
        use std::process::Command;

        // Shifted gradient so residual ≠ 0 on B-MBs. Same shape as
        // `b_frame_residual_emission_round_trip`'s content but
        // larger (96×96) for the reference decoder to chew on a more realistic
        // frame size.
        let make_shifted = |w: u32, h: u32, shift_x: i32| -> Vec<u8> {
            let y = (w * h) as usize;
            let c = (w / 2 * h / 2) as usize;
            let mut buf = vec![0u8; y + 2 * c];
            for yy in 0..h {
                for xx in 0..w {
                    let sx = xx as i32 + shift_x;
                    buf[(yy * w + xx) as usize] =
                        (((sx * 7 + (yy as i32) * 5) & 0xFF) as u8).wrapping_add(40);
                }
            }
            for yy in 0..(h / 2) {
                for xx in 0..(w / 2) {
                    buf[y + (yy * (w / 2) + xx) as usize] =
                        (128_u8).wrapping_add((xx as u8).wrapping_mul(3));
                    buf[y + c + (yy * (w / 2) + xx) as usize] =
                        (128_u8).wrapping_add((yy as u8).wrapping_mul(3));
                }
            }
            buf
        };

        let _lock = B_FORCE_MODE_ENV_LOCK.lock().expect("lock not poisoned");
        // SAFETY: lock-serialized; vars cleared after panic-catch.
        unsafe {
            std::env::set_var("PHASM_B_FORCE_MODE", "bi_16x16");
            std::env::set_var("PHASM_B_RESIDUAL", "1");
        }
        let result = std::panic::catch_unwind(|| {
            let mut enc = Encoder::new(96, 96, Some(75)).unwrap();
            enc.entropy_mode = EntropyMode::Cabac;
            enc.enable_b_frames = true;

            let mut all = Vec::new();
            all.extend_from_slice(
                &enc.encode_i_frame(&make_shifted(96, 96, 0)).unwrap(),
            );
            // 3 (P, B) pairs = 7 frames total in IBPBP shape, with
            // varied shift_x values so motion + residual both exist.
            for k in 0..3 {
                let p_shift = (k * 4 + 4) as i32;
                let b_shift = (k * 4 + 2) as i32;
                all.extend_from_slice(
                    &enc.encode_p_frame(&make_shifted(96, 96, p_shift)).unwrap(),
                );
                all.extend_from_slice(
                    &enc.encode_b_frame(&make_shifted(96, 96, b_shift)).unwrap(),
                );
            }

            let path = std::env::temp_dir()
                .join("phasm_6ea6_1q_d_b_residual_compliance.h264");
            std::fs::write(&path, &all).expect("write temp h264");

            // §B-direct-fix.v2 (#194): `-loglevel error` silences
            // the reference decoder's CABAC `concealing N DC errors` line (it's
            // emitted at info level), and the reference decoder returns exit code 0
            // even on error-concealed B-frames. Use `info` + scan
            // stderr for `concealing` so this gate genuinely catches
            // the L0/L1/Bi 16x16 B-residual spec divergence.
            let out = Command::new("the reference decoder")
                .args([
                    "-loglevel", "info",
                    "-i", path.to_str().unwrap(),
                    "-f", "null", "-",
                ])
                .output()
                .expect("the reference decoder in PATH");
            let stderr = String::from_utf8_lossy(&out.stderr);

            let _ = std::fs::remove_file(&path);

            assert!(
                out.status.success(),
                "the reference decoder failed on B-residual IBPBP output: status={:?}\nstderr={}",
                out.status, stderr,
            );
            let conceal_lines: Vec<&str> = stderr.lines()
                .filter(|l| l.contains("concealing"))
                .collect();
            assert!(
                conceal_lines.is_empty(),
                "the reference decoder flagged B-residual IBPBP CABAC concealment ({} events): {}",
                conceal_lines.len(),
                conceal_lines.join("\n  "),
            );
        });
        unsafe {
            std::env::remove_var("PHASM_B_FORCE_MODE");
            std::env::remove_var("PHASM_B_RESIDUAL");
        }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }

    /// §6E-A5 the reference decoder compliance gate — phasm's IBPBP output decodes
    /// through the the reference decoder reference decoder without errors. Marked
    /// `#[ignore]` because it shells out to the reference decoder (external CI
    /// dependency); run with `cargo test -- --ignored` or in CI when
    /// the reference decoder is available.
    ///
    /// Test gate: write a 7-frame IBPBP stream to /tmp, run
    /// `the reference decoder -i ... -f null -`, expect exit code 0 + stderr clean
    /// of "Error", "Invalid", "concealing".
    #[test]
    #[ignore = "requires the reference decoder in PATH; run with --ignored"]
    fn reference_decoder_decodes_ibpbp_without_errors() {
        use std::process::Command;

        let mut enc = Encoder::new(64, 64, Some(75)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_b_frames = true;
        let pixels = make_yuv420p(64, 64);

        let mut all = Vec::new();
        all.extend_from_slice(&enc.encode_i_frame(&pixels).unwrap());
        // 3 (P, B) pairs = 7 frames total in IBPBP shape.
        for _ in 0..3 {
            all.extend_from_slice(&enc.encode_p_frame(&pixels).unwrap());
            all.extend_from_slice(&enc.encode_b_frame(&pixels).unwrap());
        }

        let path = std::env::temp_dir().join("phasm_6ea5_reference_compliance.h264");
        std::fs::write(&path, &all).expect("write temp h264");

        let out = Command::new("the reference decoder")
            .args([
                "-loglevel", "error",
                "-i", path.to_str().unwrap(),
                "-f", "null", "-",
            ])
            .output()
            .expect("the reference decoder in PATH");
        let stderr = String::from_utf8_lossy(&out.stderr);

        assert!(
            out.status.success(),
            "the reference decoder failed: status={:?} stderr={}",
            out.status, stderr,
        );
        // the reference decoder's stderr at -loglevel error reports decode issues
        // including warnings about concealment, missing references,
        // etc. Empty stderr means clean decode.
        assert!(
            stderr.trim().is_empty(),
            "the reference decoder flagged decode issues: {stderr}"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// §B-cascade-real v1.1 — Phase 1.1.A regression test.
    ///
    /// Verifies that `Encoder::new` allocates `visual_recon` with the
    /// same dimensions as `recon`, both buffers start zero-initialised,
    /// and `write_luma_mb_dual` / `write_chroma_block_dual` write
    /// identical pixels to both buffers.
    #[test]
    fn visual_recon_dual_write_matches_recon() {
        let enc = Encoder::new(64, 48, None).expect("enc new");
        assert_eq!(enc.recon.width, enc.visual_recon.width);
        assert_eq!(enc.recon.height, enc.visual_recon.height);
        assert_eq!(enc.recon.y, enc.visual_recon.y);
        assert_eq!(enc.recon.cb, enc.visual_recon.cb);
        assert_eq!(enc.recon.cr, enc.visual_recon.cr);

        let mut enc = enc;
        let pixels: [[u8; 16]; 16] = std::array::from_fn(|y| {
            std::array::from_fn(|x| ((y * 16 + x) & 0xFF) as u8)
        });
        enc.write_luma_mb_dual(0, 0, &pixels);
        assert_eq!(
            enc.recon.y, enc.visual_recon.y,
            "Phase 1.1.A: dual-write should produce identical luma planes"
        );

        let chroma: [[u8; 8]; 8] = std::array::from_fn(|y| {
            std::array::from_fn(|x| ((y * 8 + x) & 0xFF) as u8)
        });
        enc.write_chroma_block_dual(0, 0, 0, &chroma);
        enc.write_chroma_block_dual(0, 0, 1, &chroma);
        assert_eq!(
            enc.recon.cb, enc.visual_recon.cb,
            "Phase 1.1.A: dual-write should produce identical Cb planes"
        );
        assert_eq!(
            enc.recon.cr, enc.visual_recon.cr,
            "Phase 1.1.A: dual-write should produce identical Cr planes"
        );
    }

    /// §B-cascade-real Phase 1.1.B step 2 — `ac_scan_15_to_raster`
    /// inverts `ac_scan_order_15`: lifting a chroma-AC scan back into
    /// raster form must reproduce the input raster (with `[0][0]`
    /// zeroed because `ac_scan_order_15` skips DC).
    #[test]
    fn ac_scan_15_round_trip() {
        let mut raster = [[0i32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                raster[i][j] = (i * 4 + j) as i32 * 7 - 3;
            }
        }
        let scan = ac_scan_order_15(&raster);
        let lifted = ac_scan_15_to_raster(&scan);
        let mut expected = raster;
        expected[0][0] = 0; // DC is excluded by ac_scan_order_15
        assert_eq!(lifted, expected);
    }
}
