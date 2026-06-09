// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 CABAC ctxIdxOffset tables and per-block-cat offsets.
//!
//! All direction-neutral spec data lives here. (The forward CABAC
//! encoder that previously shared these tables was removed in the
//! video-retirement; the bin-decoder `cabac::bin_decoder::syntax` is
//! now the sole consumer.) Used to compute
//! `ctxIdx = ctxIdxOffset + ctxIdxBlockCatOffset + ctxIdxInc`.
//!
//! Tabular data is spec-defined (ITU-T H.264 03/2010 Tables 9-34,
//! 9-40, 9-43); placement here is for code-organisation, not
//! semantic.

/// Per-element `ctxIdxOffset` values per spec Table 9-34.
pub mod ctx_offset {
    // mb_type / mb_skip_flag prefixes and suffixes.
    pub const MB_TYPE_SI_PREFIX: u32 = 0;
    pub const MB_TYPE_I: u32 = 3;
    pub const MB_SKIP_FLAG_P: u32 = 11;
    pub const MB_TYPE_P_PREFIX: u32 = 14;
    pub const MB_TYPE_P_SUFFIX: u32 = 17;
    pub const SUB_MB_TYPE_P: u32 = 21;
    pub const MB_SKIP_FLAG_B: u32 = 24;
    pub const MB_TYPE_B_PREFIX: u32 = 27;
    pub const MB_TYPE_B_SUFFIX: u32 = 32;
    pub const SUB_MB_TYPE_B: u32 = 36;
    pub const MVD_L0_X: u32 = 40;
    pub const MVD_L0_Y: u32 = 47;
    pub const REF_IDX: u32 = 54;
    pub const MB_QP_DELTA: u32 = 60;
    pub const INTRA_CHROMA_PRED_MODE: u32 = 64;
    pub const PREV_INTRA_PRED_MODE_FLAG: u32 = 68;
    pub const REM_INTRA_PRED_MODE: u32 = 69;
    pub const CBP_LUMA: u32 = 73;
    pub const CBP_CHROMA: u32 = 77;
    pub const CODED_BLOCK_FLAG_LOW: u32 = 85; // ctxBlockCat < 5
    pub const SIGNIFICANT_COEFF_FLAG_FRAME_LOW: u32 = 105;
    pub const LAST_SIGNIFICANT_COEFF_FLAG_FRAME_LOW: u32 = 166;
    pub const COEFF_ABS_LEVEL_MINUS1_LOW: u32 = 227;
    pub const END_OF_SLICE_FLAG: u32 = 276;
    /// `transform_size_8x8_flag` ctxIdxOffset (spec § 9.3.3.1.1.10).
    /// High-profile only — emitted when PPS `transform_8x8_mode_flag = 1`.
    pub const TRANSFORM_SIZE_8X8_FLAG: u32 = 399;
}

/// `ctxIdxBlockCatOffset` per spec Table 9-40. Indexed by
/// `[element][ctxBlockCat]` where element is:
///   0 = coded_block_flag
///   1 = significant_coeff_flag
///   2 = last_significant_coeff_flag
///   3 = coeff_abs_level_minus1
/// Covers ctxBlockCat 0..=4 (4:2:0 4×4 residuals). Cat 5
/// (Luma8×8) uses a different formula — see
/// [`ctx_offset::CAT5_LUMA8X8_SIG`] etc.
pub const CTX_BLOCK_CAT_OFFSET: [[u32; 5]; 4] = [
    [0, 4, 8, 12, 16],   // coded_block_flag
    [0, 15, 29, 44, 47], // significant_coeff_flag (frame)
    [0, 15, 29, 44, 47], // last_significant_coeff_flag (frame)
    [0, 10, 20, 30, 39], // coeff_abs_level_minus1
];

/// `significant_coeff_flag_offset_8x8[frame]` from spec Table 9-43
/// "Mapping of scanning position to ctxIdxInc for ctxBlockCat = 5,
/// 9, or 13" (ITU-T H.264 03/2010, page 270), frame-coded row.
/// Indexed by `levelListIdx` 0..=62. Cat 5 luma 8×8 residual uses
/// `sig_ctx = CAT5_LUMA8X8_SIG_FRAME +
///  SIG_COEFF_FLAG_OFFSET_8X8_FRAME[levelListIdx]`.
///
/// Tabular data is spec-defined (non-copyrightable).
pub const SIG_COEFF_FLAG_OFFSET_8X8_FRAME: [u8; 63] = [
    0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9, 8,
    7, 7, 6, 11, 12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11, 12, 13, 11, 6, 9, 14, 10, 9, 11, 12,
    13, 11, 14, 10, 12,
];

/// `last_significant_coeff_flag_offset_8x8[frame]` from spec
/// Table 9-43 (ITU-T H.264 03/2010, page 270), frame-coded row.
/// Cat 5 last_coeff_flag uses `last_ctx = CAT5_LUMA8X8_LAST_FRAME +
/// LAST_COEFF_FLAG_OFFSET_8X8_FRAME[levelListIdx]`.
///
/// Tabular data is spec-defined (non-copyrightable).
pub const LAST_COEFF_FLAG_OFFSET_8X8_FRAME: [u8; 63] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8,
];

/// Spec-derived ctxIdxOffset bases for ctxBlockCat = 5 (Luma 8×8)
/// residual emission. Per H.264 spec Table 9-34 "Syntax elements
/// and associated types of binarization, maxBinIdxCtx, and
/// ctxIdxOffset" (page 250) the relevant ctxIdxOffset values for
/// cat 5 frame-coded residuals are 402 (sig_coeff_flag), 417
/// (last_significant_coeff_flag), and 426 (coeff_abs_level_minus1).
pub mod cat5_luma8x8 {
    /// Base ctxIdx for sig_coeff_flag_8x8 (frame coding).
    pub const SIG_BASE: u32 = 402;
    /// Base ctxIdx for last_significant_coeff_flag_8x8 (frame coding).
    pub const LAST_BASE: u32 = 417;
    /// Base ctxIdx for coeff_abs_level_minus1_8x8.
    pub const ABS_BASE: u32 = 426;
}
