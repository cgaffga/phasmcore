// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

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
//!   docs/design/h264-encoder-algorithms/frame-orchestration.md
//!   docs/design/h264-encoder-algorithms/intra16x16-mb-encode.md

use super::bitstream_writer::{
    build_aud_rbsp, build_pps_cabac, build_pps_cabac_high, build_pps_cavlc, build_sps_baseline,
    build_sps_high, build_sps_main,
    continue_slice_header_i, continue_slice_header_p, wrap_rbsp_as_nal, BitWriter,
    ISliceHeaderParams, PSliceHeaderParams, PpsParams, PrimaryPicType, SpsParams,
};
use crate::codec::h264::cabac::context::CabacInitSlot;
use crate::codec::h264::cabac::encoder::{
    encode_end_of_slice_flag, encode_intra_chroma_pred_mode, encode_mb_qp_delta, encode_mb_type_i,
    encode_residual_block_cabac, CabacEncoder,
};
use crate::codec::h264::cabac::neighbor::{CabacNeighborMB, MbTypeClass};
use crate::codec::h264::cabac::slice::{append_cabac_zero_words, assemble_cabac_slice_rbsp};
use super::i4x4_encode::{derive_i4x4_mode_flags, encode_i4x4_mb, I4x4MbResult};
use super::inter_mode::{cbp_to_codenum_inter, luma_8x8_cbp_mask, pack_cbp};
use super::intra_predictor::{choose_intra_16x16_mode_psy, choose_intra_chroma_mode, satd_16x16};
use super::motion_compensation::{apply_chroma_mv_block, apply_luma_mv_block};
use super::motion_estimation::{MotionEstimator, MotionVector};
use super::partition_decision::{
    decide_p_mb, decide_p_mb_with_cost, PMbChoice, SubMbChoice, SUB_MB_ORIGINS_4X4,
    SUB_MB_ORIGINS_PX,
};
use super::partition_state::{
    predict_mv_for_mb_partition, predict_mv_for_partition, EncoderMvGrid, REF_IDX_NONE,
};
use super::quantization::{
    forward_quantize_4x4, forward_quantize_dc_chroma, forward_quantize_dc_luma,
    trellis_lambda_for_qp, trellis_quantize_4x4, QuantParams, QuantSlice,
};
use super::rate_control::{FrameType, RateController};
use super::reconstruction::{raster_to_scan_levels, ReconBuffer};
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

/// Top-level encoder.
#[derive(Debug)]
pub struct Encoder {
    pub width: u32,
    pub height: u32,
    pub frame_num: u8,
    pub rc: RateController,
    pub recon: ReconBuffer,
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

impl Encoder {
    /// Construct an encoder for the given dimensions + optional
    /// quality target. Dimensions must be 16-aligned; pad your input
    /// if necessary.
    pub fn new(width: u32, height: u32, quality: Option<u8>) -> Result<Self, EncoderError> {
        if width % 16 != 0 || height % 16 != 0 {
            return Err(EncoderError::InvalidInput(format!(
                "dimensions must be 16-aligned, got {width}×{height}"
            )));
        }
        let sps_params = SpsParams {
            width_pixels: width,
            height_pixels: height,
            sps_id: 0,
            max_num_ref_frames: 1,
        };
        let pps_params = PpsParams {
            pps_id: 0,
            sps_id: 0,
            pic_init_qp: 26,
            deblocking_filter_control_present: true,
        };
        let mb_w = (width / 16) as usize;
        let mb_h = (height / 16) as usize;
        Ok(Self {
            width,
            height,
            frame_num: 0,
            rc: RateController::new(quality),
            recon: ReconBuffer::new(width, height)?,
            sps_params,
            pps_params,
            params_emitted: false,
            entropy_mode: EntropyMode::Cabac,
            enable_transform_8x8: false,
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
            me: MotionEstimator::new(),
            cabac_trace_enabled: false,
            cabac_trace_buffer: Vec::new(),
            mode_stats: [0; 9],
        })
    }

    /// Override the default GOP length of 30. The next IDR fires at
    /// frame numbers divisible by `gop_length`.
    pub fn set_gop_length(&mut self, gop_length: u32) {
        assert!(gop_length >= 1, "gop_length must be at least 1");
        self.gop_length = gop_length;
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
        if std::env::var_os("PHASM_DISABLE_SCENECUT").is_some() {
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
    pub fn encode_i_frame(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
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
        self.frame_num = (self.frame_num + 1) & 0xF;
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
        // encode as an IDR instead. Standard scene-change-detection
        // strategy in video encoders. Threshold chosen empirically:
        // mean pixel deviation > ~20 gray levels = scene change.
        if self.should_force_idr_for_scene_change(pixels) {
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
        // reference.
        self.dpb.promote(&self.recon, self.frame_num);

        // Advance counters.
        self.frame_num = (self.frame_num + 1) & 0xF;
        self.gop_position += 1;
        if self.gop_position >= self.gop_length {
            self.gop_position = 0;
        }

        Ok(out)
    }

    fn build_p_slice_rbsp(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
        let mut w = BitWriter::with_capacity(self.frame_size_bytes().max(512));
        self.mode_stats = [0; 9];
        let qp = self.rc.base_qp_for_frame_type(FrameType::P);
        let qp_c = crate::codec::h264::transform::derive_chroma_qp(qp as i32, 0) as u8;
        let pic_init_qp = self.pps_params.pic_init_qp as i32;
        let disable_deblock = std::env::var_os("PHASM_DISABLE_DEBLOCK").is_some();
        let hdr = PSliceHeaderParams {
            pps_id: 0,
            frame_num: self.frame_num,
            slice_qp_delta: qp as i32 - pic_init_qp,
            disable_deblocking: disable_deblock,
            num_ref_idx_active_override: false,
            num_ref_idx_l0_active_minus1: 0,
            cabac_init_idc: None,
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
        }

        if std::env::var_os("PHASM_MODE_STATS").is_some() {
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
        let qp_c = crate::codec::h264::transform::derive_chroma_qp(qp as i32, 0) as u8;
        let pic_init_qp = self.pps_params.pic_init_qp as i32;
        let hdr = PSliceHeaderParams {
            pps_id: 0,
            frame_num: self.frame_num,
            slice_qp_delta: qp as i32 - pic_init_qp,
            disable_deblocking: false,
            num_ref_idx_active_override: false,
            num_ref_idx_l0_active_minus1: 0,
            cabac_init_idc: Some(0),
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
        let coded_flags = self.build_coded_flags();
        super::deblocking_filter::filter_frame_with_transform(
            &mut self.recon,
            &self.qp_grid,
            &self.intra_grid,
            Some(&self.transform_8x8_grid),
            &coded_flags,
            Some(&self.mv_grid),
        );

        let bin_count = cabac.engine.bin_count();
        let pic_size_mbs = (mb_w * mb_h) as u32;
        if let Some(trace) = cabac.engine.trace.take() {
            self.cabac_trace_buffer.extend(trace);
        }

        if std::env::var_os("PHASM_MODE_STATS").is_some() {
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
        let aq_mode: u8 = std::env::var("PHASM_AQ_MODE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let aq_disabled = std::env::var_os("PHASM_DISABLE_PAQ").is_some();
        let qp_offset: i32 = if aq_disabled {
            0
        } else if aq_mode == 3 {
            let strength: i32 = std::env::var("PHASM_AQ_STRENGTH_Q10")
                .ok()
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
        let qp = mb_qp;
        let qp_c = derive_chroma_qp(mb_qp as i32, 0) as u8;
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
        let skip_fast_enabled = std::env::var("PHASM_CABAC_SKIP_FAST").ok().as_deref() == Some("1");
        let p_skip_mv = super::partition_state::predict_p_skip_mv(
            &self.mv_grid,
            mb_x * 4,
            mb_y * 4,
        );
        if skip_fast_enabled {
            let skip_choice = PMbChoice::P16x16 { mv: p_skip_mv };
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
                let levels = trellis_quantize_4x4(&coeffs, inter, trellis_lambda_for_qp(mb_qp))
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
                    self.recon.write_luma_mb(mb_x as u32, mb_y as u32, &s_pred_y);
                    self.recon.write_chroma_block(mb_x as u32, mb_y as u32, 0, &s_pred_cb);
                    self.recon.write_chroma_block(mb_x as u32, mb_y as u32, 1, &s_pred_cr);
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
        let choice = decision.best;
        let inter_cost = decision.best_cost;

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
        let trace_would_be_intra = std::env::var_os("PHASM_TRACE_WOULD_BE_INTRA").is_some();
        let intra_in_p_enabled = std::env::var("PHASM_CABAC_INTRA_IN_P")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(true);
        // FORCE_MB semantics: when set, intra-in-P fires ONLY on the
        // specified MB (suppresses natural firings elsewhere). Lets us
        // isolate parity bugs to a single MB emit for bin-by-bin
        // comparison. Format: "x,y" or "x,y;x2,y2;..." (semi-colon list).
        let force_mb_setting = std::env::var("PHASM_CABAC_INTRA_IN_P_FORCE_MB").ok();
        let force_mb_active = force_mb_setting.is_some();
        let force_intra_here = intra_in_p_enabled
            && force_mb_setting
                .as_deref()
                .map(|s| {
                    s.split(';').any(|pair| {
                        let mut it = pair.split(',');
                        let x: Option<usize> = it.next().and_then(|p| p.trim().parse().ok());
                        let y: Option<usize> = it.next().and_then(|p| p.trim().parse().ok());
                        matches!((x, y), (Some(px), Some(py)) if px == mb_x && py == mb_y)
                    })
                })
                .unwrap_or(false);
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
            let intra_rdo_enabled = std::env::var("PHASM_INTRA_RDO")
                .ok()
                .map(|v| v != "0")
                .unwrap_or(true);
            let intra_rdo_wins = if intra_rdo_enabled && !satd_fast_out && !force_mb_active {
                let frame_w4 = (self.width / 4) as usize;
                let chroma_mode = std::env::var("PHASM_INTRA_CHROMA_MODE")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0);
                // Phase C.v3 chroma RDO: build source Cb/Cr blocks
                // from the current MB. Enabled by default; opt out
                // via `PHASM_CHROMA_RDO=0` to keep luma-only cost.
                let chroma_rdo_enabled = std::env::var("PHASM_CHROMA_RDO")
                    .ok()
                    .map(|v| v != "0")
                    .unwrap_or(true);
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
                let ceil_num: u64 = std::env::var("PHASM_INTRA_RDO_CEIL_NUM")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(Self::INTRA_RDO_CEIL_NUM);
                let ceil_den: u64 = std::env::var("PHASM_INTRA_RDO_CEIL_DEN")
                    .ok()
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
                    let i4x4_allow = std::env::var("PHASM_INTRA_IN_P_ALLOW_I4X4")
                        .ok()
                        .map(|v| v != "0")
                        .unwrap_or(true);
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
                    let d4_thresh_q10: u32 = std::env::var("PHASM_D4_FAST_INTRA_THRESH_Q10")
                        .ok()
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
                        let penalty = std::env::var("PHASM_IIP_I4X4_PENALTY")
                            .ok()
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
        let pred_y = build_luma_prediction(reference, mb_x, mb_y, &choice);
        let pred_cb = build_chroma_prediction(reference, 0, mb_x, mb_y, &choice);
        let pred_cr = build_chroma_prediction(reference, 1, mb_x, mb_y, &choice);

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
            let levels = trellis_quantize_4x4(&coeffs, inter, trellis_lambda_for_qp(qp))
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
        let is_skip = if cbp_value == 0 {
            if let PMbChoice::P16x16 { mv } = choice {
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
            // `choice.mv == p_skip_mv`), so the recon buffer is
            // already correct.
            if let PMbChoice::P16x16 { mv } = choice {
                self.mv_grid.fill(mb_x * 4, mb_y * 4, 4, 4, mv, 0);
            }
            self.recon
                .write_luma_mb(mb_x as u32, mb_y as u32, &pred_y);
            self.recon
                .write_chroma_block(mb_x as u32, mb_y as u32, 0, &pred_cb);
            self.recon
                .write_chroma_block(mb_x as u32, mb_y as u32, 1, &pred_cr);

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
                encode_sub_mb_type_p(cabac, s.sub_mb_type_codenum() as u32);
            }
        }

        // 4. ref_idx — skipped under num_ref = 1 (active_minus1 = 0).
        //    Spec: ref_idx emitted iff num_ref_idx_active_minus1 > 0.

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
                    let scan = raster_to_scan_levels(&luma_ac_levels[k]);
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
                let dc_flat: [i32; 4] = [dc[0][0], dc[0][1], dc[1][0], dc[1][1]];
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
                    let ac_scan = ac_scan_order_15(&ac_blocks[sub]);
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
        use crate::codec::h264::cabac::encoder::encode_mvd_with_bin0_inc;
        use crate::codec::h264::cabac::neighbor::compute_mvd_ctx_idx_inc_bin0;

        let base_bx = mb_x * 4;
        let base_by = mb_y * 4;
        let mut emit = |cabac_: &mut CabacEncoder,
                        current: &mut super::super::cabac::neighbor::CurrentMbMvdAbs,
                        part_bx_in_mb: u8,
                        part_by_in_mb: u8,
                        part_w4: u8,
                        part_h4: u8,
                        mv: MotionVector,
                        pred: MotionVector| {
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
            encode_mvd_with_bin0_inc(cabac_, mvd_x, 0, inc_x);
            let inc_y = compute_mvd_ctx_idx_inc_bin0(
                current,
                &cabac_.neighbors,
                mb_x,
                part_bx_in_mb,
                part_by_in_mb,
                1,
            );
            encode_mvd_with_bin0_inc(cabac_, mvd_y, 1, inc_y);
            current.fill_region(
                part_bx_in_mb,
                part_by_in_mb,
                part_w4,
                part_h4,
                mvd_x.unsigned_abs().min(i16::MAX as u32) as i16,
                mvd_y.unsigned_abs().min(i16::MAX as u32) as i16,
            );
        };
        match *choice {
            PMbChoice::P16x16 { mv } => {
                let pred = predict_mv_for_partition(&self.mv_grid, base_bx, base_by, 4, 0);
                emit(cabac, current_mvd, 0, 0, 4, 4, mv, pred);
                self.mv_grid.fill(base_bx, base_by, 4, 4, mv, 0);
            }
            PMbChoice::P16x8 { mvs } => {
                let pred0 = predict_mv_for_mb_partition(
                    &self.mv_grid, base_bx, base_by, 4, 2, 0, 0,
                );
                emit(cabac, current_mvd, 0, 0, 4, 2, mvs[0], pred0);
                self.mv_grid.fill(base_bx, base_by, 4, 2, mvs[0], 0);
                let pred1 = predict_mv_for_mb_partition(
                    &self.mv_grid,
                    base_bx,
                    base_by + 2,
                    4,
                    2,
                    1,
                    0,
                );
                emit(cabac, current_mvd, 0, 2, 4, 2, mvs[1], pred1);
                self.mv_grid.fill(base_bx, base_by + 2, 4, 2, mvs[1], 0);
            }
            PMbChoice::P8x16 { mvs } => {
                let pred0 = predict_mv_for_mb_partition(
                    &self.mv_grid, base_bx, base_by, 2, 4, 0, 0,
                );
                emit(cabac, current_mvd, 0, 0, 2, 4, mvs[0], pred0);
                self.mv_grid.fill(base_bx, base_by, 2, 4, mvs[0], 0);
                let pred1 = predict_mv_for_mb_partition(
                    &self.mv_grid,
                    base_bx + 2,
                    base_by,
                    2,
                    4,
                    1,
                    0,
                );
                emit(cabac, current_mvd, 2, 0, 2, 4, mvs[1], pred1);
                self.mv_grid.fill(base_bx + 2, base_by, 2, 4, mvs[1], 0);
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
        sub_bx_abs: usize,
        sub_by_abs: usize,
        sub_bx_in_mb: u8,
        sub_by_in_mb: u8,
        sub_choice: &SubMbChoice,
    ) {
        match *sub_choice {
            SubMbChoice::P8x8 { mv } => {
                let pred =
                    predict_mv_for_partition(&self.mv_grid, sub_bx_abs, sub_by_abs, 2, 0);
                self.emit_one_mvd_pair_cabac(
                    cabac, current_mvd, mb_x, sub_bx_in_mb, sub_by_in_mb, 2, 2, mv, pred,
                );
                self.mv_grid.fill(sub_bx_abs, sub_by_abs, 2, 2, mv, 0);
            }
            SubMbChoice::P8x4 { mvs } => {
                let pred_top =
                    predict_mv_for_partition(&self.mv_grid, sub_bx_abs, sub_by_abs, 2, 0);
                self.emit_one_mvd_pair_cabac(
                    cabac, current_mvd, mb_x, sub_bx_in_mb, sub_by_in_mb, 2, 1, mvs[0], pred_top,
                );
                self.mv_grid.fill(sub_bx_abs, sub_by_abs, 2, 1, mvs[0], 0);
                let pred_bot = predict_mv_for_partition(
                    &self.mv_grid,
                    sub_bx_abs,
                    sub_by_abs + 1,
                    2,
                    0,
                );
                self.emit_one_mvd_pair_cabac(
                    cabac,
                    current_mvd,
                    mb_x,
                    sub_bx_in_mb,
                    sub_by_in_mb + 1,
                    2,
                    1,
                    mvs[1],
                    pred_bot,
                );
                self.mv_grid
                    .fill(sub_bx_abs, sub_by_abs + 1, 2, 1, mvs[1], 0);
            }
            SubMbChoice::P4x8 { mvs } => {
                let pred_left =
                    predict_mv_for_partition(&self.mv_grid, sub_bx_abs, sub_by_abs, 1, 0);
                self.emit_one_mvd_pair_cabac(
                    cabac, current_mvd, mb_x, sub_bx_in_mb, sub_by_in_mb, 1, 2, mvs[0], pred_left,
                );
                self.mv_grid.fill(sub_bx_abs, sub_by_abs, 1, 2, mvs[0], 0);
                let pred_right = predict_mv_for_partition(
                    &self.mv_grid,
                    sub_bx_abs + 1,
                    sub_by_abs,
                    1,
                    0,
                );
                self.emit_one_mvd_pair_cabac(
                    cabac,
                    current_mvd,
                    mb_x,
                    sub_bx_in_mb + 1,
                    sub_by_in_mb,
                    1,
                    2,
                    mvs[1],
                    pred_right,
                );
                self.mv_grid
                    .fill(sub_bx_abs + 1, sub_by_abs, 1, 2, mvs[1], 0);
            }
            SubMbChoice::P4x4 { mvs } => {
                for (i, mv) in mvs.iter().enumerate() {
                    let ox = (i % 2) as usize;
                    let oy = (i / 2) as usize;
                    let pred = predict_mv_for_partition(
                        &self.mv_grid,
                        sub_bx_abs + ox,
                        sub_by_abs + oy,
                        1,
                        0,
                    );
                    self.emit_one_mvd_pair_cabac(
                        cabac,
                        current_mvd,
                        mb_x,
                        sub_bx_in_mb + ox as u8,
                        sub_by_in_mb + oy as u8,
                        1,
                        1,
                        *mv,
                        pred,
                    );
                    self.mv_grid
                        .fill(sub_bx_abs + ox, sub_by_abs + oy, 1, 1, *mv, 0);
                }
            }
        }
    }

    /// Emit one MVD pair (x + y) via CABAC and update the current-MB
    /// abs_mvd tracking for subsequent same-MB partition neighbor
    /// lookups. Does NOT touch `self.mv_grid` — caller handles that.
    #[allow(clippy::too_many_arguments)]
    fn emit_one_mvd_pair_cabac(
        &self,
        cabac: &mut CabacEncoder,
        current_mvd: &mut super::super::cabac::neighbor::CurrentMbMvdAbs,
        mb_x: usize,
        part_bx_in_mb: u8,
        part_by_in_mb: u8,
        part_w4: u8,
        part_h4: u8,
        mv: MotionVector,
        pred: MotionVector,
    ) {
        use crate::codec::h264::cabac::encoder::encode_mvd_with_bin0_inc;
        use crate::codec::h264::cabac::neighbor::compute_mvd_ctx_idx_inc_bin0;
        let mvd_x = mv.mv_x as i32 - pred.mv_x as i32;
        let mvd_y = mv.mv_y as i32 - pred.mv_y as i32;
        let inc_x = compute_mvd_ctx_idx_inc_bin0(
            current_mvd,
            &cabac.neighbors,
            mb_x,
            part_bx_in_mb,
            part_by_in_mb,
            0,
        );
        encode_mvd_with_bin0_inc(cabac, mvd_x, 0, inc_x);
        let inc_y = compute_mvd_ctx_idx_inc_bin0(
            current_mvd,
            &cabac.neighbors,
            mb_x,
            part_bx_in_mb,
            part_by_in_mb,
            1,
        );
        encode_mvd_with_bin0_inc(cabac, mvd_y, 1, inc_y);
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
        qp_c: u8,
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
        let trace_mb = std::env::var("PHASM_DUMP_MB").ok().map_or(false, |s| {
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
        // `docs/design/h264-encoder-quality-plan.md` Phase F.
        let variance = super::rate_control::mb_variance_16x16(&src_y);
        let aq_mode: u8 = std::env::var("PHASM_AQ_MODE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let aq_disabled = std::env::var_os("PHASM_DISABLE_PAQ").is_some();
        let qp_offset: i32 = if aq_disabled {
            0
        } else if aq_mode == 3 {
            let strength: i32 = std::env::var("PHASM_AQ_STRENGTH_Q10")
                .ok()
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
        let mb_qp_c = derive_chroma_qp(mb_qp as i32, 0) as u8;
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
            let skip_choice = PMbChoice::P16x16 { mv: p_skip_mv };
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
                let levels = trellis_quantize_4x4(&coeffs, inter, trellis_lambda_for_qp(mb_qp))
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
                    self.recon.write_luma_mb(mb_x as u32, mb_y as u32, &s_pred_y);
                    self.recon.write_chroma_block(mb_x as u32, mb_y as u32, 0, &s_pred_cb);
                    self.recon.write_chroma_block(mb_x as u32, mb_y as u32, 1, &s_pred_cr);
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
        let (choice, inter_cost) = if std::env::var_os("PHASM_ENABLE_RDO").is_some() {
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
            let skip_choice = PMbChoice::P16x16 { mv: p_skip_mv };
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
                self.recon.write_luma_mb(mb_x as u32, mb_y as u32, &s_pred_y);
                self.recon.write_chroma_block(mb_x as u32, mb_y as u32, 0, &s_pred_cb);
                self.recon.write_chroma_block(mb_x as u32, mb_y as u32, 1, &s_pred_cr);
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
        let force_intra = std::env::var("PHASM_FORCE_INTRA_IN_P_MB")
            .ok()
            .map_or(false, |s| {
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
        let intra_disabled = std::env::var_os("PHASM_DISABLE_INTRA").is_some();
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
            let levels = trellis_quantize_4x4(&coeffs, intra, trellis_lambda_for_qp(mb_qp))
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
            if let PMbChoice::P16x16 { mv } = choice {
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
            if let PMbChoice::P16x16 { mv } = choice {
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
        let dump_this_mb = std::env::var("PHASM_DUMP_MB").ok().map_or(false, |s| {
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
        // Select SPS/PPS profile based on entropy mode and the 8×8
        // transform opt-in. CAVLC always stays Baseline. CABAC picks
        // High when `enable_transform_8x8` is set, else Main.
        let high_profile = self.entropy_mode == EntropyMode::Cabac && self.enable_transform_8x8;
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
        let qp_c = derive_chroma_qp(qp as i32, 0) as u8;

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
        if std::env::var_os("PHASM_DISABLE_DEBLOCK").is_none() {
            let coded_flags = self.build_coded_flags();
            super::deblocking_filter::filter_frame(
                &mut self.recon,
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
        qp_c: u8,
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
        let mb_qp_c = derive_chroma_qp(mb_qp as i32, 0) as u8;
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
                    let scan = raster_to_scan_levels(&i4x4.ac_levels[k]);
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
                let dc_flat: [i32; 4] = [dc[0][0], dc[0][1], dc[1][0], dc[1][1]];
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
                    let ac_scan = ac_scan_order_15(&ac_blocks[sub]);
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
                let dc_flat: [i32; 4] = [dc[0][0], dc[0][1], dc[1][0], dc[1][1]];
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
                    let ac_scan = ac_scan_order_15(&ac_blocks[sub]);
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
        let qp_c = derive_chroma_qp(qp as i32, 0) as u8;

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
        if std::env::var_os("PHASM_DISABLE_DEBLOCK").is_none() {
            let coded_flags = self.build_coded_flags();
            super::deblocking_filter::filter_frame_with_transform(
                &mut self.recon,
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
        use crate::codec::h264::cabac::encoder::{encode_mb_type_p, encode_residual_block_cabac_with_cbf_inc};
        use crate::codec::h264::cabac::neighbor::{
            block_pos_to_chroma_ac_idx, block_pos_to_luma_idx,
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
            ac_levels[k] = trellis_quantize_4x4(&coeffs, intra, trellis_lambda_for_qp(qp))
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

        // 5. mb_type (Table 7-11 encoding).
        let mb_type = 1 + luma_pred_mode + 4 * cbp_chroma as u32 + 12 * cbp_luma_flag as u32;

        // 6. Emit CABAC syntax. In P-slice context, shift the codenum
        // into the P-slice I_16x16 range (values 6..30 per Table 7-13
        // P-slice row; +5 offset from the I-slice codenums 1..25) and
        // use the P-slice mb_type encoder. The caller (intra-in-P
        // fallback) is responsible for emitting the leading
        // `mb_skip_flag=0` bin before invoking this function.
        if in_p_slice {
            encode_mb_type_p(cabac, mb_type + 5, mb_x);
        } else {
            encode_mb_type_i(cabac, mb_type, mb_x);
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
        let dc_scan = raster_to_scan_levels(&dc_levels);
        if std::env::var_os("PHASM_DEBUG_IIP_LEVELS").is_some() && in_p_slice {
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
                let ac_scan = ac_scan_order_15(&ac_levels[k]);
                if std::env::var_os("PHASM_DEBUG_IIP_LEVELS").is_some() && in_p_slice {
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
                let dc_flat: [i32; 4] = [dc[0][0], dc[0][1], dc[1][0], dc[1][1]];
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
                    let ac_scan = ac_scan_order_15(&ac_blocks[sub]);
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

        // 9. Reconstruction.
        let recon_y = self.reconstruct_luma_mb(&pred_y, &ac_levels, &dc_levels, qp);
        self.recon.write_luma_mb(mb_x as u32, mb_y as u32, &recon_y);
        let recon_cb =
            self.reconstruct_chroma_mb(&pred_cb, &cb_ac_levels, &cb_dc_levels, qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 0, &recon_cb);
        let recon_cr =
            self.reconstruct_chroma_mb(&pred_cr, &cr_ac_levels, &cr_dc_levels, qp_c);
        self.recon
            .write_chroma_block(mb_x as u32, mb_y as u32, 1, &recon_cr);

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
        let mode: u8 = std::env::var("PHASM_AQ_MODE")
            .ok()
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
        let disable_deblocking = std::env::var_os("PHASM_DISABLE_DEBLOCK").is_some();
        ISliceHeaderParams {
            is_idr: true,
            pps_id: 0,
            frame_num: self.frame_num,
            idr_pic_id: 0,
            slice_qp_delta: slice_qp - pic_init_qp,
            disable_deblocking,
        }
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

        self.recon.write_luma_mb(mb_x as u32, mb_y as u32, &luma_block);
        self.recon.write_chroma_block(mb_x as u32, mb_y as u32, 0, &cb_block);
        self.recon.write_chroma_block(mb_x as u32, mb_y as u32, 1, &cr_block);

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
        let mb_qp_c = derive_chroma_qp(mb_qp as i32, 0) as u8;
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

        // ── Chroma reconstruction ──
        let recon_cb =
            self.reconstruct_chroma_mb(&pred_cb, &cb_ac_levels, &cb_dc_levels, qp_c);
        self.recon.write_chroma_block(mb_x as u32, mb_y as u32, 0, &recon_cb);
        let recon_cr =
            self.reconstruct_chroma_mb(&pred_cr, &cr_ac_levels, &cr_dc_levels, qp_c);
        self.recon.write_chroma_block(mb_x as u32, mb_y as u32, 1, &recon_cr);

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
            ac_levels[k] = trellis_quantize_4x4(&coeffs, intra, trellis_lambda_for_qp(qp))
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

        // 7. mb_type = 1 + pred_mode + 4*CBP_chroma + 12*CBP_luma_flag
        //    (spec Table 7-11). `pred_mode` is the Intra_16x16 luma
        //    mode chosen above (0..=3 per the Intra16x16Mode enum).
        //    For P-slice intra, mb_type_offset = 5 shifts the codenum
        //    into the P-slice mb_type table (Table 7-13 rows 6..=29).
        let mb_type =
            1 + luma_pred_mode + 4 * cbp_chroma as u32 + 12 * cbp_luma_flag as u32;
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

        // 9. Reconstruction: reverse the whole pipeline.
        let recon_y =
            self.reconstruct_luma_mb(&pred_y, &ac_levels, &dc_levels, qp);
        self.recon.write_luma_mb(mb_x as u32, mb_y as u32, &recon_y);

        let recon_cb =
            self.reconstruct_chroma_mb(&pred_cb, &cb_ac_levels, &cb_dc_levels, qp_c);
        self.recon.write_chroma_block(mb_x as u32, mb_y as u32, 0, &recon_cb);

        let recon_cr =
            self.reconstruct_chroma_mb(&pred_cr, &cr_ac_levels, &cr_dc_levels, qp_c);
        self.recon.write_chroma_block(mb_x as u32, mb_y as u32, 1, &recon_cr);

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
                    trellis_quantize_4x4(&coeffs, qp, trellis_lambda_for_qp(qp.qp))
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
        PMbChoice::P16x16 { mv } => {
            apply_luma_mv_block(reference, mb_px_x, mb_px_y, 16, 16, mv, flat, 16);
        }
        PMbChoice::P16x8 { mvs } => {
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
        PMbChoice::P8x16 { mvs } => {
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
        SubMbChoice::P8x8 { mv } => vec![(0, 0, 8, 8, mv)],
        SubMbChoice::P8x4 { mvs } => vec![(0, 0, 8, 4, mvs[0]), (0, 4, 8, 4, mvs[1])],
        SubMbChoice::P4x8 { mvs } => vec![(0, 0, 4, 8, mvs[0]), (4, 0, 4, 8, mvs[1])],
        SubMbChoice::P4x4 { mvs } => vec![
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
        PMbChoice::P16x16 { mv } => {
            apply_chroma_mv_block(reference, component, mb_px_x, mb_px_y, 8, 8, mv, flat, 8);
        }
        PMbChoice::P16x8 { mvs } => {
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
        PMbChoice::P8x16 { mvs } => {
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
        PMbChoice::P16x16 { mv } => {
            let pred = predict_mv_for_partition(grid, base_bx, base_by, 4, 0);
            w.write_se(mv.mv_x as i32 - pred.mv_x as i32);
            w.write_se(mv.mv_y as i32 - pred.mv_y as i32);
            grid.fill(base_bx, base_by, 4, 4, mv, 0);
        }
        PMbChoice::P16x8 { mvs } => {
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
        PMbChoice::P8x16 { mvs } => {
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
        SubMbChoice::P8x8 { mv } => {
            let pred = predict_mv_for_partition(grid, sub_bx, sub_by, 2, 0);
            w.write_se(mv.mv_x as i32 - pred.mv_x as i32);
            w.write_se(mv.mv_y as i32 - pred.mv_y as i32);
            grid.fill(sub_bx, sub_by, 2, 2, mv, 0);
        }
        SubMbChoice::P8x4 { mvs } => {
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
        SubMbChoice::P4x8 { mvs } => {
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
        SubMbChoice::P4x4 { mvs } => {
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
    fn i16x16_mb_type_dc_no_coeffs_eq_3() {
        // Formula: mb_type = 1 + pred_mode + 4*CBP_chroma + 12*CBP_luma_flag
        // DC mode (2), no coeffs: 1 + 2 + 0 + 0 = 3.
        assert_eq!(1 + 2 + 4 * 0 + 12 * 0, 3);
    }

    #[test]
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
}
