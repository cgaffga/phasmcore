// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 macroblock parsing with neighbor context tracking for nC computation.
//!
//! Parses macroblock types (I_4x4, I_16x16, P_Skip, P_16x16, etc.), prediction
//! modes, coded block pattern, and dispatches to the CAVLC decoder for each
//! 4x4 residual block. Maintains a 2D grid of TotalCoeffs for nC neighbor context.
//!
//! Reference: ITU-T H.264 Sections 7.3.5 (macroblock layer) and 7.4.5.

use super::bitstream::{EpByteMap, RbspReader};
use super::cavlc::{decode_cavlc_block, CavlcBlock, EmbeddablePosition};
use super::slice::SliceType;
use super::sps::{Pps, Sps};
use super::H264Error;

/// H.264 macroblock type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbType {
    /// I_4x4: intra prediction with 4x4 block modes.
    I4x4,
    /// I_16x16: intra prediction with 16x16 mode.
    /// Contains (intra_16x16_pred_mode, cbp_luma, cbp_chroma).
    I16x16(u8, u8, u8),
    /// I_PCM: raw uncompressed samples (no transform/quant).
    IPCM,
    /// P_Skip: no residual, motion derived from neighbors.
    PSkip,
    /// P_L0_16x16: single 16x16 inter partition.
    P16x16,
    /// P_L0_L0_16x8: two 16x8 inter partitions.
    P16x8,
    /// P_L0_L0_8x16: two 8x16 inter partitions.
    P8x16,
    /// P_8x8: four 8x8 sub-macroblock partitions.
    P8x8,
    /// P_8x8ref0: 8x8 with ref_idx forced to 0.
    P8x8ref0,
}

impl MbType {
    pub fn is_intra(self) -> bool {
        matches!(self, Self::I4x4 | Self::I16x16(_, _, _) | Self::IPCM)
    }

    pub fn is_skip(self) -> bool {
        matches!(self, Self::PSkip)
    }
}

/// Decoded macroblock with embeddable positions.
#[derive(Debug, Clone)]
pub struct Macroblock {
    pub mb_type: MbType,
    pub mb_qp_delta: i32,
    pub coded_block_pattern: u32,
    /// TotalCoeffs for each 4x4 luma block (16 blocks in raster order).
    pub luma_total_coeffs: [u8; 16],
    /// TotalCoeffs for each 4x4 chroma block (4 Cb + 4 Cr = 8).
    pub chroma_total_coeffs: [u8; 8],
    /// Embeddable positions found in this macroblock.
    pub positions: Vec<EmbeddablePosition>,
    /// Pixel-reconstruction data, populated when `capture_recon = true` and
    /// the MB is intra (I_4x4 / I_16x16). Used by Phase 1b UNIWARD cost to
    /// rebuild Y-plane pixels for a wavelet decomposition. `None` for P/B MBs,
    /// I_PCM (samples not kept), and when the caller opts out.
    pub recon: Option<ReconstructionData>,
    /// Motion vector field for P-slice MBs (list-0 only; Baseline/Main).
    /// `None` for intra / skip / PCM. Phase 2 populates this via
    /// `mv::parse_mv_field`; Phase 3 and later DDCA use it.
    pub mv_field: Option<super::mv::MvField>,
}

/// Reconstruction data for an intra macroblock — enough to produce 16×16
/// Y samples plus 8×8 Cb/Cr samples (4:2:0) via `transform.rs` +
/// `intra_pred.rs`.
#[derive(Debug, Clone)]
pub struct ReconstructionData {
    /// Luma QP of this MB after `mb_qp_delta` is applied. Drives luma dequant.
    pub qp_y: i32,
    /// Chroma QPs derived from `qp_y` and the PPS chroma QP offsets. Both are
    /// pre-computed at parse time to save the caller a table lookup.
    pub qp_cb: i32,
    pub qp_cr: i32,
    /// For I_4x4 MBs: the 16 resolved intra-prediction modes (H.264 spec
    /// 8.3.1.1), one per 4×4 block in raster order. Values are 0..=8 matching
    /// `intra_pred::Intra4x4Mode`. Zero for non-I_4x4 MBs.
    pub intra4x4_modes: [u8; 16],
    /// For I_16x16 MBs: the macroblock-level prediction mode (0..=3 per
    /// `intra_pred::Intra16x16Mode`). `None` for I_4x4 / other.
    pub intra16x16_mode: Option<u8>,
    /// `intra_chroma_pred_mode` (0..=3 per `intra_pred::IntraChroma8x8Mode`).
    /// Applies to both Cb and Cr.
    pub intra_chroma_pred_mode: u8,
    /// Residual coefficients per 4×4 luma block, in scan (zigzag) order. For
    /// I_16x16 the `[0]` entry of each AC block is the placeholder zero; the
    /// real DC values come from `luma_dc_block` via the Hadamard path.
    pub luma_blocks: [[i32; 16]; 16],
    /// Intra_16x16 DC coefficients (16 scan-order values). `None` for I_4x4.
    pub luma_dc_block: Option<[i32; 16]>,
    /// Chroma DC coefficients in scan order (4 for 4:2:0). Index 0 = Cb, 1 = Cr.
    /// Populated only when `cbp_chroma > 0`.
    pub chroma_dc_blocks: [Option<[i32; 4]>; 2],
    /// Chroma AC coefficients per 4×4 block, scan order. Layout:
    /// `[component][block_idx]` where component 0 = Cb, 1 = Cr, block_idx
    /// 0..=3 in raster order inside the 8×8 chroma block. Only populated
    /// when `cbp_chroma == 2`.
    pub chroma_ac_blocks: [[[i32; 16]; 4]; 2],
}

impl ReconstructionData {
    /// Construct a zero-initialised reconstruction block with the given luma QP.
    /// Chroma QPs default to the same value (intended for tests).
    pub fn empty(qp_y: i32) -> Self {
        Self {
            qp_y,
            qp_cb: qp_y,
            qp_cr: qp_y,
            intra4x4_modes: [0; 16],
            intra16x16_mode: None,
            intra_chroma_pred_mode: 0,
            luma_blocks: [[0i32; 16]; 16],
            luma_dc_block: None,
            chroma_dc_blocks: [None, None],
            chroma_ac_blocks: [[[0i32; 16]; 4]; 2],
        }
    }
}

/// Neighbor context for nC (TotalCoeffs) computation across macroblocks.
///
/// Maintains a grid of TotalCoeffs per 4x4 block for the current and
/// previous macroblock rows. Each macroblock has 16 luma 4x4 blocks
/// (4 columns x 4 rows) and 8 chroma blocks (2x2 per Cb/Cr).
pub struct NeighborContext {
    width_in_mbs: u32,
    /// TotalCoeffs for luma 4x4 blocks, layout: [mb_y * width_in_mbs * 16 + mb_x * 4 + block_col]
    /// for the top row of each macroblock (needed by the MB below).
    /// We store the full frame's 4x4 block grid for simplicity.
    /// Index: (mb_y * 4 + block_row) * (width_in_mbs * 4) + mb_x * 4 + block_col
    luma_tc: Vec<u8>,
    luma_width: usize, // width_in_mbs * 4
    /// Chroma TotalCoeffs (similar layout but 2x2 blocks per MB).
    chroma_cb_tc: Vec<u8>,
    chroma_cr_tc: Vec<u8>,
    chroma_width: usize, // width_in_mbs * 2
    /// I_16x16 DC block TotalCoeffs, one per macroblock.
    /// Used for nC computation of the current MB's Intra16x16DCLevel block.
    /// Stored as 0xFF (invalid) when MB is not I_16x16.
    luma_dc_tc: Vec<u8>,
    /// Resolved Intra_4x4 prediction mode per 4x4 luma block (0..=8).
    /// Stored as 0xFF for blocks that aren't I_4x4 (treated as "not available"
    /// by the neighbor-based predicted-mode derivation in H.264 8.3.1.1).
    intra4x4_modes: Vec<u8>,
}

impl NeighborContext {
    pub fn new(width_in_mbs: u32, height_in_mbs: u32) -> Self {
        let luma_width = (width_in_mbs * 4) as usize;
        let luma_height = (height_in_mbs * 4) as usize;
        let chroma_width = (width_in_mbs * 2) as usize;
        let chroma_height = (height_in_mbs * 2) as usize;
        let mb_count = (width_in_mbs * height_in_mbs) as usize;
        Self {
            width_in_mbs,
            luma_tc: vec![0; luma_width * luma_height],
            luma_width,
            chroma_cb_tc: vec![0; chroma_width * chroma_height],
            chroma_cr_tc: vec![0; chroma_width * chroma_height],
            chroma_width,
            // 0xFF marks "not an I_16x16 MB" — treated as unavailable for nC.
            luma_dc_tc: vec![0xFF; mb_count],
            // 0xFF marks "not an I_4x4 block" — per spec 8.3.1.1 this is
            // "predIntra4x4PredMode not available" → fallback to DC (mode 2).
            intra4x4_modes: vec![0xFF; luma_width * luma_height],
        }
    }

    /// Store the resolved Intra_4x4 prediction mode for a 4x4 luma block.
    pub fn set_intra4x4_mode(&mut self, block_x: usize, block_y: usize, mode: u8) {
        let idx = block_y * self.luma_width + block_x;
        if idx < self.intra4x4_modes.len() {
            self.intra4x4_modes[idx] = mode;
        }
    }

    /// Read the Intra_4x4 prediction mode at a 4x4 luma block. Returns `None`
    /// when the position is out-of-range or has never been set (e.g., top edge
    /// of the frame, or a non-I_4x4 MB).
    pub fn intra4x4_mode(&self, block_x: usize, block_y: usize) -> Option<u8> {
        let idx = block_y * self.luma_width + block_x;
        self.intra4x4_modes
            .get(idx)
            .copied()
            .filter(|&v| v != 0xFF)
    }

    /// Store Intra16x16DCLevel TotalCoeffs for the macroblock at (mb_x, mb_y).
    pub fn set_luma_dc_tc(&mut self, mb_x: u32, mb_y: u32, tc: u8) {
        let idx = (mb_y * self.width_in_mbs + mb_x) as usize;
        if idx < self.luma_dc_tc.len() {
            self.luma_dc_tc[idx] = tc;
        }
    }

    /// Get nC for the Intra16x16DCLevel block of the MB at (mb_x, mb_y).
    ///
    /// Per H.264 § 9.2.1.1: nC for Intra16x16DC is derived "as if
    /// Luma4x4 with luma4x4BlkIdx = 0 were being decoded". So
    /// neighbors are the LEFT and ABOVE 4x4 LUMA blocks (i.e., the
    /// rightmost-column / bottommost-row 4x4 blocks of the
    /// neighbor MBs) — NOT a DC TC. Equivalent to `luma_nc(mb_x*4,
    /// mb_y*4)`.
    pub fn luma_dc_nc(&self, mb_x: u32, mb_y: u32) -> i8 {
        let block_x = (mb_x * 4) as usize;
        let block_y = (mb_y * 4) as usize;
        self.luma_nc(block_x, block_y)
    }

    /// Store TotalCoeffs for a luma 4x4 block at (4x4 grid position).
    pub fn set_luma_tc(&mut self, block_x: usize, block_y: usize, tc: u8) {
        let idx = block_y * self.luma_width + block_x;
        if idx < self.luma_tc.len() {
            self.luma_tc[idx] = tc;
        }
    }

    /// Get nC for a luma 4x4 block at (4x4 grid position).
    /// nC = (nA + nB + 1) >> 1 where nA=left, nB=above.
    /// At edges: if left unavailable, nC=nB; if above unavailable, nC=nA; if both, nC=0.
    pub fn luma_nc(&self, block_x: usize, block_y: usize) -> i8 {
        let have_left = block_x > 0;
        let have_above = block_y > 0;

        match (have_left, have_above) {
            (false, false) => 0,
            (true, false) => self.luma_tc[block_y * self.luma_width + block_x - 1] as i8,
            (false, true) => {
                self.luma_tc[(block_y - 1) * self.luma_width + block_x] as i8
            }
            (true, true) => {
                let na = self.luma_tc[block_y * self.luma_width + block_x - 1] as i16;
                let nb =
                    self.luma_tc[(block_y - 1) * self.luma_width + block_x] as i16;
                ((na + nb + 1) >> 1) as i8
            }
        }
    }

    /// Get nC for a chroma 4x4 block (Cb or Cr).
    pub fn chroma_nc(&self, block_x: usize, block_y: usize, is_cr: bool) -> i8 {
        let tc = if is_cr {
            &self.chroma_cr_tc
        } else {
            &self.chroma_cb_tc
        };
        let have_left = block_x > 0;
        let have_above = block_y > 0;

        match (have_left, have_above) {
            (false, false) => 0,
            (true, false) => tc[block_y * self.chroma_width + block_x - 1] as i8,
            (false, true) => tc[(block_y - 1) * self.chroma_width + block_x] as i8,
            (true, true) => {
                let na = tc[block_y * self.chroma_width + block_x - 1] as i16;
                let nb = tc[(block_y - 1) * self.chroma_width + block_x] as i16;
                ((na + nb + 1) >> 1) as i8
            }
        }
    }

    /// Store TotalCoeffs for a chroma 4x4 block.
    pub fn set_chroma_tc(
        &mut self,
        block_x: usize,
        block_y: usize,
        is_cr: bool,
        tc: u8,
    ) {
        let grid = if is_cr {
            &mut self.chroma_cr_tc
        } else {
            &mut self.chroma_cb_tc
        };
        let idx = block_y * self.chroma_width + block_x;
        if idx < grid.len() {
            grid[idx] = tc;
        }
    }
}

/// 4x4 block raster-scan index within a macroblock → (col, row) within MB.
/// H.264 scans 4x4 luma blocks in raster order within each 8x8 block,
/// and the four 8x8 blocks in raster order.
/// Block index 0-3: top-left 8x8, Block index 4-7: top-right 8x8, etc.
pub const BLOCK_INDEX_TO_POS: [(u8, u8); 16] = [
    (0, 0), (1, 0), (0, 1), (1, 1), // 8x8 block 0 (top-left)
    (2, 0), (3, 0), (2, 1), (3, 1), // 8x8 block 1 (top-right)
    (0, 2), (1, 2), (0, 3), (1, 3), // 8x8 block 2 (bottom-left)
    (2, 2), (3, 2), (2, 3), (3, 3), // 8x8 block 3 (bottom-right)
];

/// CBP (coded block pattern) luma/chroma from mb_type for I_16x16.
/// mb_type 1-24 for I_16x16 decodes to (pred_mode, cbp_luma, cbp_chroma)
/// per H.264 Table 7-11. Mapping:
/// - pred_mode = (mb_type-1) % 4 ∈ {0,1,2,3}
/// - group = (mb_type-1) / 4 ∈ {0..5}
/// - cbp_chroma = group % 3 ∈ {0,1,2}
/// - cbp_luma = 0 if group < 3 else 15
///
/// The type name `I_16x16_<pred>_<CBPLuma>_<CBPChroma>` in the spec lists
/// the fields in order pred, CBPLuma_flag (0=none, 1=all), CBPChroma.
fn decode_i16x16_mb_type(mb_type_minus1: u32) -> (u8, u8, u8) {
    let pred_mode = (mb_type_minus1 % 4) as u8;
    let group = mb_type_minus1 / 4;
    let cbp_chroma = (group % 3) as u8;
    let cbp_luma = if group < 3 { 0 } else { 15 };
    (pred_mode, cbp_luma, cbp_chroma)
}

/// Coded block pattern mapping for inter macroblocks (Table 9-4b).
/// Maps code_num from ue(v) to (cbp_luma, cbp_chroma_coded).
const CBP_INTER_TABLE: [u32; 48] = [
    0,  16, 1,  2,  4,  8,  32, 3,  5,  10, 12, 15, 47, 7,  11, 13,
    14, 6,  9,  31, 35, 37, 42, 44, 33, 34, 36, 40, 39, 43, 45, 46,
    17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

/// Coded block pattern mapping for intra macroblocks (Table 9-4a).
pub const CBP_INTRA_TABLE: [u32; 48] = [
    47, 31, 15, 0,  23, 27, 29, 30, 7,  11, 13, 14, 39, 43, 45, 46,
    16, 3,  5,  10, 12, 19, 21, 26, 28, 35, 37, 42, 44, 1,  2,  4,
    8,  17, 18, 20, 24, 6,  9,  22, 25, 32, 33, 34, 36, 40, 38, 41,
];

/// Parse a macroblock and extract embeddable CAVLC positions.
///
/// This is the main entry point for macroblock-level parsing. It dispatches
/// to the CAVLC decoder for each 4x4 residual block and tracks nC context.
pub fn parse_macroblock(
    reader: &mut RbspReader<'_>,
    slice_type: SliceType,
    mb_x: u32,
    mb_y: u32,
    sps: &Sps,
    pps: &Pps,
    ctx: &mut NeighborContext,
    ep_map: &EpByteMap,
    raw_data: &[u8],
    current_qp: &mut i32,
    num_ref_idx_l0_active: u8,
) -> Result<Macroblock, H264Error> {
    parse_macroblock_with_recon(
        reader,
        slice_type,
        mb_x,
        mb_y,
        sps,
        pps,
        ctx,
        ep_map,
        raw_data,
        current_qp,
        num_ref_idx_l0_active,
        false,
        None,
    )
}

/// Variant of [`parse_macroblock`] that optionally captures pixel-reconstruction
/// data in `Macroblock::recon`. Set `capture_recon = true` on intra slices to
/// enable Phase 1b UNIWARD cost computation; pass `false` on P/B slices and
/// when you only need embeddable positions (Phase 1a).
///
/// Phase 2: if `mv_ctx` is `Some`, P-slice macroblocks will have their motion
/// vectors parsed and stored in `Macroblock::mv_field`, and the frame-wide
/// `MvPredictorContext` will be updated so subsequent MBs can use the stored
/// MVs for their own median predictors. Pass `None` to keep the Phase 1a
/// behaviour (MVDs are discarded).
pub fn parse_macroblock_with_recon(
    reader: &mut RbspReader<'_>,
    slice_type: SliceType,
    mb_x: u32,
    mb_y: u32,
    sps: &Sps,
    pps: &Pps,
    ctx: &mut NeighborContext,
    ep_map: &EpByteMap,
    raw_data: &[u8],
    current_qp: &mut i32,
    num_ref_idx_l0_active: u8,
    capture_recon: bool,
    mv_ctx: Option<&mut super::mv::MvPredictorContext>,
) -> Result<Macroblock, H264Error> {
    let mut positions = Vec::new();
    let mut luma_total_coeffs = [0u8; 16];
    let mut chroma_total_coeffs = [0u8; 8];
    let mut recon: Option<ReconstructionData> = None;

    // Parse mb_type
    let mb_type = parse_mb_type(reader, slice_type)?;

    if mb_type == MbType::PSkip {
        // Skip macroblock: no residual, no coded data. For P_Skip the spec
        // uses a simplified "inferred" MV (zero or a derived predictor); we
        // record a zero MV here so downstream consumers still see a real
        // MvField. A more accurate P_Skip MV derivation can land later.
        if let Some(ctx) = mv_ctx {
            let base_x = (mb_x * 4) as usize;
            let base_y = (mb_y * 4) as usize;
            for by in 0..4 {
                for bx in 0..4 {
                    ctx.set(base_x + bx, base_y + by, super::mv::MotionVector::default(), 0);
                }
            }
        }
        return Ok(Macroblock {
            mb_type,
            mb_qp_delta: 0,
            coded_block_pattern: 0,
            luma_total_coeffs,
            chroma_total_coeffs,
            positions,
            recon: None,
            mv_field: Some(super::mv::MvField::default()),
        });
    }

    if mb_type == MbType::IPCM {
        // I_PCM: skip raw sample bytes (256 luma + 2*64 chroma for 4:2:0)
        reader.align_to_byte();
        let pcm_bytes = 256 + 2 * 64; // 4:2:0
        reader.skip_bits(pcm_bytes * 8)?;
        // All TotalCoeffs = 16 for I_PCM (per spec)
        for tc in &mut luma_total_coeffs {
            *tc = 16;
        }
        for tc in &mut chroma_total_coeffs {
            *tc = 16;
        }
        update_neighbor_tc(ctx, mb_x, mb_y, &luma_total_coeffs, &chroma_total_coeffs);
        return Ok(Macroblock {
            mb_type,
            mb_qp_delta: 0,
            coded_block_pattern: 0,
            luma_total_coeffs,
            chroma_total_coeffs,
            positions,
            // I_PCM reconstruction (raw samples) is deferred; UNIWARD callers
            // can treat these MBs as a 128-fill or skip them.
            recon: None,
            mv_field: None,
        });
    }

    // Initialise `recon` lazily for intra MBs when capture is requested. We
    // fill in QP after mb_qp_delta is applied below.
    if capture_recon && mb_type.is_intra() {
        recon = Some(ReconstructionData::empty(*current_qp));
    }

    // Parse prediction info. For I_4x4 this now resolves and stores the 16
    // per-block intra modes in the neighbor context (needed for both neighbor
    // mode prediction of later MBs and for reconstruction). For I_16x16 the
    // mode is already encoded in `mb_type`; we pull it into `recon` below.
    // For P-slice MBs, if mv_ctx is Some, this also parses and stores MVs
    // (Phase 2) and captures MVD suffix-LSB embeddable positions into
    // `positions` (Phase 3a).
    let parsed_mv_field = parse_prediction_info(
        reader,
        mb_type,
        slice_type,
        sps,
        pps,
        num_ref_idx_l0_active,
        mb_x,
        mb_y,
        ctx,
        recon.as_mut(),
        mv_ctx,
        ep_map,
        raw_data,
        &mut positions,
    )?;

    // Parse coded_block_pattern (except I_16x16 where it's in mb_type)
    let coded_block_pattern = match mb_type {
        MbType::I16x16(_, cbp_luma, cbp_chroma) => {
            (cbp_chroma as u32) << 4 | cbp_luma as u32
        }
        _ => {
            let cbp_code = reader.read_ue()?;
            if cbp_code >= 48 {
                return Err(H264Error::CavlcError(format!(
                    "cbp code_num {cbp_code} >= 48"
                )));
            }
            if mb_type.is_intra() {
                CBP_INTRA_TABLE[cbp_code as usize]
            } else {
                CBP_INTER_TABLE[cbp_code as usize]
            }
        }
    };

    let cbp_luma = coded_block_pattern & 0x0F;
    let cbp_chroma = (coded_block_pattern >> 4) & 0x03;

    // Parse mb_qp_delta (if any coded residual).
    // Per H.264 Section 7.4.5: QP wraps modulo 52 for 8-bit video.
    // Spec: QPy = ((QPy_prev + mb_qp_delta + 52) % 52)
    let mb_qp_delta = if cbp_luma > 0 || cbp_chroma > 0 || matches!(mb_type, MbType::I16x16(_, _, _)) {
        let delta = reader.read_se()?;
        *current_qp = ((*current_qp + delta).rem_euclid(52) + 52) % 52;
        delta
    } else {
        0
    };

    // Refresh QP in the (lazily-initialised) recon block.
    if let Some(r) = recon.as_mut() {
        r.qp_y = *current_qp;
        r.qp_cb = super::transform::derive_chroma_qp(*current_qp, pps.chroma_qp_index_offset);
        r.qp_cr = super::transform::derive_chroma_qp(
            *current_qp,
            pps.second_chroma_qp_index_offset,
        );
    }

    // Parse residual data (CAVLC blocks)
    let base_luma_x = (mb_x * 4) as usize;
    let base_luma_y = (mb_y * 4) as usize;

    if let MbType::I16x16(pred_mode, _, _) = mb_type {
        // I_16x16: first decode Intra16x16DCLevel (one 4x4 DC block, max_coeffs=16).
        let nc = ctx.luma_dc_nc(mb_x, mb_y);
        let (dc_block, mut dc_positions) =
            decode_cavlc_block(reader, nc, ep_map, raw_data, 16)?;
        // I_16x16 luma DC doesn't occupy a 26-slot per-MB block_idx; positions
        // in the Hadamard-transformed DC block carry high impact and are not
        // the target of Phase 1b/2 embedding. Mark with the u32::MAX sentinel
        // so cost functions return f32::INFINITY.
        for p in &mut dc_positions {
            p.block_idx = u32::MAX;
        }
        positions.extend(dc_positions);

        // Store this MB's DC TotalCoeffs for neighbor lookups.
        ctx.set_luma_dc_tc(mb_x, mb_y, dc_block.total_coeffs);

        if let Some(r) = recon.as_mut() {
            r.intra16x16_mode = Some(pred_mode);
            r.luma_dc_block = Some(dc_block.coeffs);
        }

        // Then decode Intra16x16ACLevel for each of the 16 luma blocks
        // (15 AC coefficients per block, max_coeffs=15)
        if cbp_luma != 0 {
            for blk_idx in 0..16u8 {
                let (bx, by) = BLOCK_INDEX_TO_POS[blk_idx as usize];
                let luma_4x4_x = base_luma_x + bx as usize;
                let luma_4x4_y = base_luma_y + by as usize;
                let nc = ctx.luma_nc(luma_4x4_x, luma_4x4_y);

                let (block, mut block_positions) =
                    decode_cavlc_block(reader, nc, ep_map, raw_data, 15)?;
                // Luma AC slot = block index in BLOCK_INDEX_TO_POS order (0..15).
                for p in &mut block_positions {
                    p.block_idx = blk_idx as u32;
                }
                luma_total_coeffs[blk_idx as usize] = block.total_coeffs;
                ctx.set_luma_tc(luma_4x4_x, luma_4x4_y, block.total_coeffs);
                positions.extend(block_positions);

                if let Some(r) = recon.as_mut() {
                    r.luma_blocks[blk_idx as usize] = block.coeffs;
                }
            }
        } else {
            // cbp_luma=0: no AC coefficients coded. For neighbor nC, the AC
            // TotalCoeffs is 0 (only DC block existed, not used for AC nC).
            for blk_idx in 0..16u8 {
                let (bx, by) = BLOCK_INDEX_TO_POS[blk_idx as usize];
                ctx.set_luma_tc(
                    base_luma_x + bx as usize,
                    base_luma_y + by as usize,
                    0,
                );
            }
        }
    } else {
        // I_4x4, P_16x16, etc.: decode 16 luma 4x4 blocks
        for blk_idx in 0..16u8 {
            let (bx, by) = BLOCK_INDEX_TO_POS[blk_idx as usize];
            let luma_4x4_x = base_luma_x + bx as usize;
            let luma_4x4_y = base_luma_y + by as usize;

            // Check if this 8x8 block has coded coefficients (cbp_luma is per-8x8)
            let blk8x8_idx = (by / 2) * 2 + bx / 2;
            if cbp_luma & (1 << blk8x8_idx) != 0 {
                let nc = ctx.luma_nc(luma_4x4_x, luma_4x4_y);
                let (block, mut block_positions) =
                    decode_cavlc_block(reader, nc, ep_map, raw_data, 16)?;
                // Luma slot = block index 0..15.
                for p in &mut block_positions {
                    p.block_idx = blk_idx as u32;
                }
                luma_total_coeffs[blk_idx as usize] = block.total_coeffs;
                ctx.set_luma_tc(luma_4x4_x, luma_4x4_y, block.total_coeffs);
                positions.extend(block_positions);

                if let Some(r) = recon.as_mut() {
                    r.luma_blocks[blk_idx as usize] = block.coeffs;
                }
            } else {
                ctx.set_luma_tc(luma_4x4_x, luma_4x4_y, 0);
            }
        }
    }

    // Parse chroma residual (4:2:0: 4 Cb + 4 Cr blocks)
    let base_chroma_x = (mb_x * 2) as usize;
    let base_chroma_y = (mb_y * 2) as usize;

    if cbp_chroma > 0 {
        // Chroma DC (2x2 blocks). Slot 16 for Cb, 17 for Cr.
        for is_cr in [false, true] {
            let (dc_block, mut dc_positions) =
                decode_cavlc_block(reader, -1, ep_map, raw_data, 4)?;
            let slot: u32 = if is_cr { 17 } else { 16 };
            for p in &mut dc_positions {
                p.block_idx = slot;
            }
            positions.extend(dc_positions);
            if let Some(r) = recon.as_mut() {
                let mut dc = [0i32; 4];
                dc.copy_from_slice(&dc_block.coeffs[..4]);
                r.chroma_dc_blocks[if is_cr { 1 } else { 0 }] = Some(dc);
            }
        }

        // Chroma AC (4 blocks per component, if cbp_chroma == 2).
        // Slot 18..21 for Cb, 22..25 for Cr.
        if cbp_chroma == 2 {
            for is_cr in [false, true] {
                for blk_idx in 0..4u8 {
                    let cx = base_chroma_x + (blk_idx % 2) as usize;
                    let cy = base_chroma_y + (blk_idx / 2) as usize;
                    let nc = ctx.chroma_nc(cx, cy, is_cr);

                    let (block, mut block_positions) =
                        decode_cavlc_block(reader, nc, ep_map, raw_data, 15)?;
                    let slot: u32 = if is_cr { 22 } else { 18 } + blk_idx as u32;
                    for p in &mut block_positions {
                        p.block_idx = slot;
                    }
                    let chroma_offset = if is_cr { 4 } else { 0 };
                    chroma_total_coeffs[chroma_offset + blk_idx as usize] =
                        block.total_coeffs;
                    ctx.set_chroma_tc(cx, cy, is_cr, block.total_coeffs);
                    positions.extend(block_positions);

                    if let Some(r) = recon.as_mut() {
                        r.chroma_ac_blocks[if is_cr { 1 } else { 0 }][blk_idx as usize] =
                            block.coeffs;
                    }
                }
            }
        } else {
            // cbp_chroma == 1: only DC, no AC → AC TotalCoeffs = 0 for neighbors
            for is_cr in [false, true] {
                for blk_idx in 0..4u8 {
                    let cx = base_chroma_x + (blk_idx % 2) as usize;
                    let cy = base_chroma_y + (blk_idx / 2) as usize;
                    ctx.set_chroma_tc(cx, cy, is_cr, 0);
                }
            }
        }
    } else {
        // No chroma residual at all → AC TotalCoeffs = 0 for neighbors
        for is_cr in [false, true] {
            for blk_idx in 0..4u8 {
                let cx = base_chroma_x + (blk_idx % 2) as usize;
                let cy = base_chroma_y + (blk_idx / 2) as usize;
                ctx.set_chroma_tc(cx, cy, is_cr, 0);
            }
        }
    }

    Ok(Macroblock {
        mb_type,
        mb_qp_delta,
        coded_block_pattern,
        luma_total_coeffs,
        chroma_total_coeffs,
        positions,
        recon,
        mv_field: parsed_mv_field,
    })
}

/// Update neighbor context with this macroblock's TotalCoeffs.
fn update_neighbor_tc(
    ctx: &mut NeighborContext,
    mb_x: u32,
    mb_y: u32,
    luma_tc: &[u8; 16],
    chroma_tc: &[u8; 8],
) {
    let base_luma_x = (mb_x * 4) as usize;
    let base_luma_y = (mb_y * 4) as usize;
    for blk_idx in 0..16 {
        let (bx, by) = BLOCK_INDEX_TO_POS[blk_idx];
        ctx.set_luma_tc(
            base_luma_x + bx as usize,
            base_luma_y + by as usize,
            luma_tc[blk_idx],
        );
    }
    let base_cx = (mb_x * 2) as usize;
    let base_cy = (mb_y * 2) as usize;
    for i in 0..4 {
        ctx.set_chroma_tc(base_cx + i % 2, base_cy + i / 2, false, chroma_tc[i]);
        ctx.set_chroma_tc(base_cx + i % 2, base_cy + i / 2, true, chroma_tc[i + 4]);
    }
}

/// Parse mb_type from the bitstream.
fn parse_mb_type(
    reader: &mut RbspReader<'_>,
    slice_type: SliceType,
) -> Result<MbType, H264Error> {
    let raw = reader.read_ue()?;

    match slice_type {
        SliceType::I | SliceType::SI => {
            // I-slice mb_type (Table 7-11)
            match raw {
                0 => Ok(MbType::I4x4),
                1..=24 => {
                    let (pred, cbp_l, cbp_c) = decode_i16x16_mb_type(raw - 1);
                    Ok(MbType::I16x16(pred, cbp_l, cbp_c))
                }
                25 => Ok(MbType::IPCM),
                _ => Err(H264Error::CavlcError(format!(
                    "invalid I-slice mb_type: {raw}"
                ))),
            }
        }
        SliceType::P | SliceType::SP => {
            // P-slice mb_type (Table 7-13)
            // Note: P_Skip is signaled by mb_skip_run, not mb_type.
            // When we get here, mb_type has already been read.
            match raw {
                0 => Ok(MbType::P16x16),
                1 => Ok(MbType::P16x8),
                2 => Ok(MbType::P8x16),
                3 => Ok(MbType::P8x8),
                4 => Ok(MbType::P8x8ref0),
                // 5+ maps to I-slice types (intra in P-slice)
                5 => Ok(MbType::I4x4),
                6..=29 => {
                    let (pred, cbp_l, cbp_c) = decode_i16x16_mb_type(raw - 6);
                    Ok(MbType::I16x16(pred, cbp_l, cbp_c))
                }
                30 => Ok(MbType::IPCM),
                _ => Err(H264Error::CavlcError(format!(
                    "invalid P-slice mb_type: {raw}"
                ))),
            }
        }
        _ => Err(H264Error::Unsupported(format!(
            "B/SI slice mb_type parsing not implemented"
        ))),
    }
}

/// Parse prediction info: intra modes, motion vectors, sub-partition types.
///
/// For I_4x4 this resolves all 16 per-block intra-prediction modes (spec
/// 8.3.1.1), publishes them into `ctx.intra4x4_modes` (so neighbouring MBs
/// can use them in their own mode prediction), and — if `recon` is present —
/// records them in `recon.intra4x4_modes`.
///
/// `num_ref_idx_l0_active` is from the slice header (may override PPS default).
/// ref_idx uses te(v): 1-bit FLC when max=1, ue(v) otherwise.
fn parse_prediction_info(
    reader: &mut RbspReader<'_>,
    mb_type: MbType,
    _slice_type: SliceType,
    _sps: &Sps,
    _pps: &Pps,
    num_ref_idx_l0_active: u8,
    mb_x: u32,
    mb_y: u32,
    ctx: &mut NeighborContext,
    mut recon: Option<&mut ReconstructionData>,
    mv_ctx: Option<&mut super::mv::MvPredictorContext>,
    ep_map: &EpByteMap,
    raw_data: &[u8],
    mvd_positions: &mut Vec<EmbeddablePosition>,
) -> Result<Option<super::mv::MvField>, H264Error> {
    let max_ref = num_ref_idx_l0_active.saturating_sub(1) as u32;

    match mb_type {
        MbType::I4x4 => {
            // Per H.264 Section 8.3.1.1: for each 4x4 block in decoding order,
            // derive `predIntra4x4PredMode = min(left_mode, top_mode)` with a
            // fallback of DC (mode 2) when either neighbour is unavailable.
            // Then read the bitstream-signalled override.
            let base_x = (mb_x * 4) as usize;
            let base_y = (mb_y * 4) as usize;
            for blk_idx in 0..16usize {
                let (bx, by) = BLOCK_INDEX_TO_POS[blk_idx];
                let block_x = base_x + bx as usize;
                let block_y = base_y + by as usize;

                // Spec § 8.3.1.1:
                //   1. If mbAddrN is unavailable (frame/slice edge) →
                //      whole derivation returns 2.
                //   2. Else if neighbor N is I_4x4 → use its per-block
                //      mode; else per-neighbor mode = 2. Then
                //      predMode = min(A, B).
                //
                // `ctx.intra4x4_mode` returns None for BOTH "outside
                // the grid" AND "available but not I_4x4". To match
                // the spec we distinguish those cases via the block
                // coordinate (block_x == 0 / block_y == 0 means
                // frame-edge, i.e. mbAddrN unavailable for the edge
                // 4×4 blocks).
                let left = if bx > 0 || block_x > 0 {
                    Some(ctx.intra4x4_mode(block_x - 1, block_y).unwrap_or(2))
                } else {
                    None // mbAddrA unavailable (leftmost column of frame)
                };
                let top = if by > 0 || block_y > 0 {
                    Some(ctx.intra4x4_mode(block_x, block_y - 1).unwrap_or(2))
                } else {
                    None // mbAddrB unavailable (top row of frame)
                };
                let pred_mode = match (left, top) {
                    (Some(a), Some(b)) => a.min(b),
                    _ => 2,
                };

                let prev_flag = reader.read_bit()?;
                let resolved = if prev_flag {
                    pred_mode
                } else {
                    let rem = reader.read_bits(3)? as u8;
                    if rem < pred_mode {
                        rem
                    } else {
                        rem + 1
                    }
                };

                ctx.set_intra4x4_mode(block_x, block_y, resolved);
                if let Some(r) = recon.as_deref_mut() {
                    r.intra4x4_modes[blk_idx] = resolved;
                }
            }
            let chroma_mode = reader.read_ue()? as u8; // intra_chroma_pred_mode
            if let Some(r) = recon.as_deref_mut() {
                r.intra_chroma_pred_mode = chroma_mode;
            }
        }
        MbType::I16x16(_, _, _) => {
            let chroma_mode = reader.read_ue()? as u8; // intra_chroma_pred_mode
            if let Some(r) = recon.as_deref_mut() {
                r.intra_chroma_pred_mode = chroma_mode;
            }
        }
        MbType::P16x16 | MbType::P16x8 | MbType::P8x16 | MbType::P8x8 | MbType::P8x8ref0 => {
            // Phase 2: if an MV context is provided, parse and store MVs
            // properly. Otherwise fall back to the old discard-the-bits
            // behaviour so Phase 1a callers keep working.
            if let Some(mv_ctx) = mv_ctx {
                return Ok(super::mv::parse_mv_field(
                    reader,
                    mb_type,
                    mb_x,
                    mb_y,
                    num_ref_idx_l0_active,
                    mv_ctx,
                    ep_map,
                    raw_data,
                    mvd_positions,
                )?);
            }
            // No mv_ctx: consume the bits so the stream stays in sync but
            // discard the values (same as pre-Phase-2 behaviour).
            discard_p_slice_prediction_bits(reader, mb_type, max_ref)?;
        }
        MbType::IPCM | MbType::PSkip => {}
    }
    Ok(None)
}

/// Consume the ref_idx / MVD bits for a P-slice MB without storing anything.
/// Used when the caller hasn't opted into Phase 2 MV tracking.
fn discard_p_slice_prediction_bits(
    reader: &mut RbspReader<'_>,
    mb_type: MbType,
    max_ref: u32,
) -> Result<(), H264Error> {
    match mb_type {
        MbType::P16x16 => {
            if max_ref > 0 {
                let _ = reader.read_te(max_ref)?;
            }
            let _ = reader.read_se()?;
            let _ = reader.read_se()?;
        }
        MbType::P16x8 | MbType::P8x16 => {
            for _ in 0..2 {
                if max_ref > 0 {
                    let _ = reader.read_te(max_ref)?;
                }
            }
            for _ in 0..2 {
                let _ = reader.read_se()?;
                let _ = reader.read_se()?;
            }
        }
        MbType::P8x8 | MbType::P8x8ref0 => {
            let mut sub_types = [0u32; 4];
            for item in sub_types.iter_mut() {
                *item = reader.read_ue()?;
            }
            if mb_type != MbType::P8x8ref0 && max_ref > 0 {
                for _ in 0..4 {
                    let _ = reader.read_te(max_ref)?;
                }
            }
            for &st in &sub_types {
                let num_sub_parts = match st {
                    0 => 1,
                    1 | 2 => 2,
                    3 => 4,
                    _ => {
                        return Err(H264Error::CavlcError(format!(
                            "invalid P-slice sub_mb_type: {st}"
                        )));
                    }
                };
                for _ in 0..num_sub_parts {
                    let _ = reader.read_se()?;
                    let _ = reader.read_se()?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbor_context_basic() {
        let mut ctx = NeighborContext::new(2, 2); // 2x2 MBs = 8x8 4x4-block grid

        // Set some TotalCoeffs
        ctx.set_luma_tc(0, 0, 3);
        ctx.set_luma_tc(1, 0, 5);
        ctx.set_luma_tc(0, 1, 7);

        // Top-left corner: no neighbors → nC=0
        assert_eq!(ctx.luma_nc(0, 0), 0);
        // (1,0): left=3, above=none → nC=3
        assert_eq!(ctx.luma_nc(1, 0), 3);
        // (0,1): left=none, above=3 → nC=3
        assert_eq!(ctx.luma_nc(0, 1), 3);
        // (1,1): left=7, above=5 → nC=(7+5+1)/2=6
        assert_eq!(ctx.luma_nc(1, 1), 6);
    }

    #[test]
    fn block_index_to_pos_coverage() {
        // All 16 positions should be unique
        let mut seen = std::collections::HashSet::new();
        for &(x, y) in &BLOCK_INDEX_TO_POS {
            assert!(x < 4 && y < 4);
            assert!(seen.insert((x, y)));
        }
        assert_eq!(seen.len(), 16);
    }

    #[test]
    fn i16x16_mb_type_decode() {
        // H.264 Table 7-11: mb_type values 1-24 for I_16x16.
        // Grouped by CBP: group = (mb_type-1)/4, cbp_chroma = group%3,
        // cbp_luma = 0 if group<3 else 15, pred_mode = (mb_type-1)%4.
        //
        // Groups:
        //   group 0 (mb_type 1-4):   cbp_luma=0,  cbp_chroma=0
        //   group 1 (mb_type 5-8):   cbp_luma=0,  cbp_chroma=1
        //   group 2 (mb_type 9-12):  cbp_luma=0,  cbp_chroma=2
        //   group 3 (mb_type 13-16): cbp_luma=15, cbp_chroma=0
        //   group 4 (mb_type 17-20): cbp_luma=15, cbp_chroma=1
        //   group 5 (mb_type 21-24): cbp_luma=15, cbp_chroma=2

        // mb_type 1 → pred=0, cbp_l=0, cbp_c=0
        assert_eq!(decode_i16x16_mb_type(0), (0, 0, 0));

        // mb_type 5 → pred=0, cbp_l=0, cbp_c=1
        assert_eq!(decode_i16x16_mb_type(4), (0, 0, 1));

        // mb_type 7 → pred=2, cbp_l=0, cbp_c=1 (the case that triggered the bug)
        assert_eq!(decode_i16x16_mb_type(6), (2, 0, 1));

        // mb_type 9 → pred=0, cbp_l=0, cbp_c=2
        assert_eq!(decode_i16x16_mb_type(8), (0, 0, 2));

        // mb_type 13 → pred=0, cbp_l=15, cbp_c=0
        assert_eq!(decode_i16x16_mb_type(12), (0, 15, 0));

        // mb_type 17 → pred=0, cbp_l=15, cbp_c=1
        assert_eq!(decode_i16x16_mb_type(16), (0, 15, 1));

        // mb_type 21 → pred=0, cbp_l=15, cbp_c=2
        assert_eq!(decode_i16x16_mb_type(20), (0, 15, 2));

        // mb_type 24 → pred=3, cbp_l=15, cbp_c=2
        assert_eq!(decode_i16x16_mb_type(23), (3, 15, 2));
    }

    #[test]
    fn mb_type_properties() {
        assert!(MbType::I4x4.is_intra());
        assert!(MbType::I16x16(0, 0, 0).is_intra());
        assert!(MbType::IPCM.is_intra());
        assert!(!MbType::P16x16.is_intra());
        assert!(MbType::PSkip.is_skip());
        assert!(!MbType::P16x16.is_skip());
    }
}
