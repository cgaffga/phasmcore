// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! CABAC neighbor context + `ctxIdxInc` derivation (spec § 9.3.3.1.1).
//!
//! Each CABAC syntax element computes its own `ctxIdxInc` based on
//! the element kind (binIdx) plus state from neighboring macroblocks
//! ("A" = left, "B" = top). This module holds the per-MB state the
//! derivation rules read and the rules themselves.
//!
//! Phase 6C.4 scope: all Baseline / Main (I + P) syntax elements with
//! `ChromaArrayType = 1` (4:2:0) and progressive (non-MBAFF) coding.
//! B-slice, MBAFF, and 4:4:4 rules are stubbed with
//! `todo!()` guards — Phase 6C doesn't emit them.

use super::context::CabacInitSlot;

/// Macroblock type class for CABAC neighbor-state queries. Simplified
/// from our full `MbType` enum — only the distinctions the CABAC
/// context-derivation rules care about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbTypeClass {
    /// I_NxN (I_4x4 or I_8x8 — the two sub-types of `mb_type = 0`).
    INxN,
    /// Any I_16x16 variant (`mb_type` 1..24).
    I16x16,
    /// I_PCM (`mb_type = 25`).
    IPCM,
    /// SI slice's SI macroblock type.
    SI,
    /// P_L0_16x16 / P_L0_L0_16x8 / P_L0_L0_8x16 / P_8x8 / P_8x8ref0.
    PInter,
    /// P_Skip (signaled via `mb_skip_flag = 1`).
    PSkip,
    /// B-slice direct/skip — not used in Phase 6C.
    BSkipOrDirect,
    /// B-slice inter — not used in Phase 6C.
    BInter,
}

impl MbTypeClass {
    pub fn is_intra(self) -> bool {
        matches!(
            self,
            MbTypeClass::INxN | MbTypeClass::I16x16 | MbTypeClass::IPCM | MbTypeClass::SI
        )
    }

    pub fn is_inter(self) -> bool {
        matches!(
            self,
            MbTypeClass::PInter
                | MbTypeClass::PSkip
                | MbTypeClass::BSkipOrDirect
                | MbTypeClass::BInter
        )
    }

    pub fn is_skip(self) -> bool {
        matches!(self, MbTypeClass::PSkip | MbTypeClass::BSkipOrDirect)
    }
}

/// Per-MB state the CABAC context-derivation rules read.
///
/// This is a superset of the data the encoder already tracks in
/// `i4x4_mode_grid`, `total_coeff_grid`, `mv_grid`, and slice-level
/// neighbor tables — but re-packaged per-MB to match how the
/// spec's § 9.3.3.1.1 rules describe neighbors ("is mbAddrN
/// available?", "is `mb_type[mbAddrN]` P_Skip?", etc.).
#[derive(Debug, Clone, Copy)]
pub struct CabacNeighborMB {
    pub mb_type: MbTypeClass,
    pub mb_skip_flag: bool,
    pub intra_chroma_pred_mode: u8, // 0..3
    pub cbp_luma: u8,               // 4-bit
    pub cbp_chroma: u8,             // 2-bit
    pub mb_qp_delta: i32,           // emitted delta, for prevMbAddr lookup
    /// Per-4×4-block coded_block_flag for all residual categories we
    /// track. Indexed [ctxBlockCat][block_idx_in_mb]. For 4:2:0:
    ///   cat 0 (luma DC):   1 slot.
    ///   cat 1 (luma AC):   16 slots (one per 4×4).
    ///   cat 2 (luma 4x4):  16 slots.
    ///   cat 3 (chroma DC): 2 slots (Cb, Cr).
    ///   cat 4 (chroma AC): 8 slots (4 per plane × 2 planes).
    pub coded_block_flag_cat: [u16; 5],
    /// Absolute MVD values per 4x4 block, [component][block_in_mb].
    /// component 0 = x, 1 = y.
    pub abs_mvd_comp: [[i16; 16]; 2],
    /// ref_idx for list 0, per 4x4 block.
    pub ref_idx_l0: [i8; 16],
    /// Whether the neighbor MB used 8×8 transform. Used by
    /// `ctx_idx_inc_transform_size_8x8_flag` (spec § 9.3.3.1.1.10).
    /// Always false for Baseline/Main streams (the syntax element is
    /// never emitted there).
    pub transform_size_8x8_flag: bool,
}

impl Default for CabacNeighborMB {
    fn default() -> Self {
        Self {
            mb_type: MbTypeClass::INxN,
            mb_skip_flag: false,
            intra_chroma_pred_mode: 0,
            cbp_luma: 0,
            cbp_chroma: 0,
            mb_qp_delta: 0,
            coded_block_flag_cat: [0; 5],
            abs_mvd_comp: [[0; 16]; 2],
            ref_idx_l0: [0; 16],
            transform_size_8x8_flag: false,
        }
    }
}

/// Frame-wide CABAC neighbor-context row buffer. Tracks available
/// top-row MBs and the immediate-left MB so the derivation rules can
/// answer neighbor queries.
pub struct CabacNeighborContext {
    mb_width: usize,
    /// Top row: one entry per MB column (index = mb_x).
    top_row: Vec<Option<CabacNeighborMB>>,
    /// Immediate-left MB (index by current mb_x - 1).
    left: Option<CabacNeighborMB>,
    /// Previous-in-decoding-order MB (for `mb_qp_delta` ctxIdxInc rule).
    prev_mb: Option<CabacNeighborMB>,
    /// Slice type / init_idc — some rules depend on slice type.
    pub slice_slot: CabacInitSlot,
}

impl CabacNeighborContext {
    pub fn new(mb_width: usize, slice_slot: CabacInitSlot) -> Self {
        Self {
            mb_width,
            top_row: vec![None; mb_width],
            left: None,
            prev_mb: None,
            slice_slot,
        }
    }

    /// Reset at slice start. Top-row + left become "not available".
    pub fn reset(&mut self) {
        for slot in self.top_row.iter_mut() {
            *slot = None;
        }
        self.left = None;
        self.prev_mb = None;
    }

    pub fn neighbor_a(&self, _mb_x: usize) -> Option<&CabacNeighborMB> {
        self.left.as_ref()
    }

    pub fn neighbor_b(&self, mb_x: usize) -> Option<&CabacNeighborMB> {
        self.top_row.get(mb_x)?.as_ref()
    }

    pub fn prev_mb(&self) -> Option<&CabacNeighborMB> {
        self.prev_mb.as_ref()
    }

    /// Commit a completed MB's neighbor state. Called after the MB's
    /// CABAC emit finishes. The top-row[mb_x] is updated; the left
    /// slot rolls to this MB; `prev_mb` also tracks this MB.
    pub fn commit(&mut self, mb_x: usize, mb: CabacNeighborMB) {
        if mb_x < self.top_row.len() {
            self.top_row[mb_x] = Some(mb);
        }
        self.left = Some(mb);
        self.prev_mb = Some(mb);
    }

    /// At row boundary: roll `left` to `None` so the first MB of the
    /// next row has no A-neighbor. Call before encoding mb_x == 0 of
    /// a new row.
    pub fn new_row(&mut self) {
        self.left = None;
    }
}

// ─── Per-element ctxIdxInc derivation rules (spec § 9.3.3.1.1) ──

/// Compute `ctxIdxInc` for `mb_skip_flag` (spec § 9.3.3.1.1.1, eq 9-7).
/// `ctxIdxInc = condTermFlagA + condTermFlagB` where `condTermFlagN`
/// is 0 if N unavailable or `mb_skip_flag[N] == 1`, else 1.
pub fn ctx_idx_inc_mb_skip_flag(ctx: &CabacNeighborContext, mb_x: usize) -> u32 {
    let cond = |n: Option<&CabacNeighborMB>| -> u32 {
        match n {
            None => 0,
            Some(mb) if mb.mb_skip_flag => 0,
            Some(_) => 1,
        }
    };
    cond(ctx.neighbor_a(mb_x)) + cond(ctx.neighbor_b(mb_x))
}

/// Compute `ctxIdxInc` for `mb_type` bin 0 (spec § 9.3.3.1.1, Table 9-39).
///
/// Per Table 9-39, the binIdx=0 entry varies by ctxIdxOffset:
/// - **I slice** (`ctxIdxOffset = 3`): derived per § 9.3.3.1.1.3 —
///   `condTermFlagN = 0` if N unavailable OR N's mb_type is `I_NxN`,
///   else 1. `ctxIdxInc = condTermFlagA + condTermFlagB`.
/// - **SI slice** (`ctxIdxOffset = 0`): analogous — `condTermFlagN = 0`
///   if N unavailable OR N is SI, else 1.
/// - **B slice** (`ctxIdxOffset = 27`): `condTermFlagN = 0` if N is
///   `B_Skip` or `B_Direct_16x16`, else 1.
/// - **P/SP slice** (`ctxIdxOffset = 14`): Table 9-39 assigns a FIXED
///   `ctxIdxInc = 0` for binIdx=0 (no neighbor derivation). Each
///   subsequent binIdx then maps to its own fixed ctxIdxInc.
pub fn ctx_idx_inc_mb_type_bin0(
    ctx: &CabacNeighborContext,
    mb_x: usize,
    ctx_idx_offset: u32,
) -> u32 {
    // P/SP slice: Table 9-39 gives fixed ctxIdxInc=0 for binIdx=0;
    // no neighbor derivation. (Derived from conformance testing +
    // spec Table 9-39 re-read — Task #21, 2026-04-23.)
    if ctx_idx_offset == 14 {
        return 0;
    }
    let cond = |n: Option<&CabacNeighborMB>| -> u32 {
        let mb = match n {
            None => return 0,
            Some(mb) => mb,
        };
        match ctx_idx_offset {
            0 => {
                // SI slice: "N is SI" → 0, else 1.
                if mb.mb_type == MbTypeClass::SI { 0 } else { 1 }
            }
            3 => {
                // I slice: "N is I_NxN" → 0, else 1.
                if mb.mb_type == MbTypeClass::INxN { 0 } else { 1 }
            }
            27 => {
                // B slice: "N is B_Skip/B_Direct_16x16" → 0, else 1.
                if matches!(mb.mb_type, MbTypeClass::BSkipOrDirect) {
                    0
                } else {
                    1
                }
            }
            _ => 0,
        }
    };
    cond(ctx.neighbor_a(mb_x)) + cond(ctx.neighbor_b(mb_x))
}

/// Prior-bin-dependent ctxIdxInc for `mb_type` and `sub_mb_type`
/// (spec § 9.3.3.1.2, Table 9-41). Applied to bins 2+ of certain
/// `ctxIdxOffset` / binIdx combinations.
pub fn ctx_idx_inc_prior_bin(ctx_idx_offset: u32, bin_idx: u32, prior_bins: &[u8]) -> Option<u32> {
    match (ctx_idx_offset, bin_idx) {
        (3, 4) => Some(if prior_bins[3] != 0 { 5 } else { 6 }),
        (3, 5) => Some(if prior_bins[3] != 0 { 6 } else { 7 }),
        (14, 2) => Some(if prior_bins[1] != 1 { 2 } else { 3 }),
        // (17, 4) prior-bin rule verified by Phase D.0-B sweep
        // (2026-04-23): the pair {2, 3} matches spec Table 9-41 for
        // the offset=17 P-slice intra suffix bin 4 — inverting to
        // {3, 2} regresses parity catastrophically (256/256 px
        // diff). Keep as-is.
        (17, 4) => Some(if prior_bins[3] != 0 { 2 } else { 3 }),
        (27, 2) => Some(if prior_bins[1] != 0 { 4 } else { 5 }),
        (32, 4) => Some(if prior_bins[3] != 0 { 2 } else { 3 }),
        (36, 2) => Some(if prior_bins[1] != 0 { 2 } else { 3 }),
        _ => None,
    }
}

/// Compute ctxIdxInc for `mvd_lX` bin 0 (spec § 9.3.3.1.1.7 eq 9-15).
/// Sum of absolute neighbor MVDs thresholded at 3 and 32.
pub fn ctx_idx_inc_mvd_bin0(
    ctx: &CabacNeighborContext,
    mb_x: usize,
    block_idx_in_mb_a: usize,
    block_idx_in_mb_b: usize,
    component: u8,
) -> u32 {
    let abs_a = ctx
        .neighbor_a(mb_x)
        .map(|n| n.abs_mvd_comp[component as usize][block_idx_in_mb_a].unsigned_abs() as u32)
        .unwrap_or(0);
    let abs_b = ctx
        .neighbor_b(mb_x)
        .map(|n| n.abs_mvd_comp[component as usize][block_idx_in_mb_b].unsigned_abs() as u32)
        .unwrap_or(0);
    let sum = abs_a + abs_b;
    if sum < 3 {
        0
    } else if sum > 32 {
        2
    } else {
        1
    }
}

/// Compute ctxIdxInc for `ref_idx_lX` bin 0 (spec § 9.3.3.1.1.6, eq 9-14).
/// `ctxIdxInc = condTermFlagA + 2·condTermFlagB`.
pub fn ctx_idx_inc_ref_idx_bin0(
    ctx: &CabacNeighborContext,
    mb_x: usize,
    block_idx_in_mb_a: usize,
    block_idx_in_mb_b: usize,
) -> u32 {
    let cond = |n: Option<&CabacNeighborMB>, blk: usize| -> u32 {
        match n {
            None => 0,
            Some(mb) if mb.mb_type.is_skip() || mb.mb_type.is_intra() => 0,
            Some(mb) if mb.ref_idx_l0[blk] == 0 => 0,
            Some(_) => 1,
        }
    };
    cond(ctx.neighbor_a(mb_x), block_idx_in_mb_a)
        + 2 * cond(ctx.neighbor_b(mb_x), block_idx_in_mb_b)
}

/// Compute ctxIdxInc for `mb_qp_delta` bin 0 (spec § 9.3.3.1.1.5).
/// Uses `prevMbAddr`, NOT A/B. Returns 0 if the previous MB's
/// `mb_qp_delta` was 0 (or prev unavailable / is skip / PCM), else 1.
pub fn ctx_idx_inc_mb_qp_delta_bin0(ctx: &CabacNeighborContext) -> u32 {
    let Some(prev) = ctx.prev_mb() else {
        return 0;
    };
    if prev.mb_type == MbTypeClass::IPCM || prev.mb_type.is_skip() {
        return 0;
    }
    if prev.mb_qp_delta == 0 {
        0
    } else {
        1
    }
}

/// Compute ctxIdxInc for `intra_chroma_pred_mode` bin 0 (spec
/// § 9.3.3.1.1.8, eq 9-18). `ctxIdxInc = condTermFlagA + condTermFlagB`.
/// `condTermFlagN = 0` when N unavailable / is inter / is I_PCM /
/// `intra_chroma_pred_mode[N] == 0`, else 1.
pub fn ctx_idx_inc_intra_chroma_pred_mode_bin0(
    ctx: &CabacNeighborContext,
    mb_x: usize,
) -> u32 {
    let cond = |n: Option<&CabacNeighborMB>| -> u32 {
        match n {
            None => 0,
            Some(mb) if mb.mb_type.is_inter() || mb.mb_type == MbTypeClass::IPCM => 0,
            Some(mb) if mb.intra_chroma_pred_mode == 0 => 0,
            Some(_) => 1,
        }
    };
    cond(ctx.neighbor_a(mb_x)) + cond(ctx.neighbor_b(mb_x))
}

/// Compute ctxIdxInc for `coded_block_pattern` luma bin (spec
/// § 9.3.3.1.1.4, eq 9-10). `binIdx` refers to which 8x8 luma block's
/// CBP bit we're emitting; the neighbor-condition terms look up the
/// matching 8x8 block in each neighbor.
pub fn ctx_idx_inc_cbp_luma(
    ctx: &CabacNeighborContext,
    mb_x: usize,
    bin_idx: u32,
) -> u32 {
    // Legacy cross-MB-only form — kept for backwards compatibility with
    // older callers. For I_4x4 / P / B encoding, use
    // `compute_cbp_luma_ctx_idx_inc_bin` which correctly handles the
    // same-MB 8×8 neighbors for binIdx 1..3.
    let cross_neighbor_bit = bin_idx;
    let cond = |n: Option<&CabacNeighborMB>| cbp_luma_cross_cond(n, cross_neighbor_bit);
    cond(ctx.neighbor_a(mb_x)) + 2 * cond(ctx.neighbor_b(mb_x))
}

#[inline]
fn cbp_luma_cross_cond(n: Option<&CabacNeighborMB>, neighbor_bit: u32) -> u32 {
    // Spec § 9.3.3.1.1.4 condTermFlag for `coded_block_pattern` luma
    // bins: condTermFlag = 0 when the corresponding neighbor luma
    // cbp bit is 1, else 1. IPCM neighbors report condTermFlag = 1
    // unconditionally per the § 9.3.3.1.1.4 fallback rule. Unavailable
    // neighbors (`None`) use condTermFlag = 0.
    //
    // Historical note: an earlier version treated `is_skip()` MBs as
    // condTermFlag = 0 unconditionally, which diverged from spec
    // for P-skip neighbors because cbp_luma = 0 on a P-skip MB is
    // already captured by the general "bit is 0 → condTermFlag = 1"
    // rule — the extra skip special-case was producing a wrong
    // context in cbp_luma bin 0 for any P-MB with a P-skip left
    // neighbour.
    match n {
        None => 0,
        Some(mb) if mb.mb_type == MbTypeClass::IPCM => 1,
        Some(mb) => {
            if mb.cbp_luma & (1u8 << neighbor_bit) != 0 {
                0
            } else {
                1
            }
        }
    }
}

/// Compute ctxIdxInc for `coded_block_pattern`'s luma bins (spec
/// § 9.3.3.1.1.4 eq 9-10) with proper 8×8-block neighbor derivation.
///
/// `bin_idx` is the binIdx (0..=3) = raster-scan index of the current
/// 8×8 block within the MB. `current_partial_cbp_luma` is the
/// progressively-built 4-bit current-MB CBP luma, containing bits for
/// `bin_idx < bin_idx` that have already been emitted in this MB.
///
/// For binIdx > 0 some neighbors fall INSIDE the current MB and the
/// caller must fold their bits into `current_partial_cbp_luma` before
/// each call. Cross-MB neighbors use `neighbors.neighbor_a/b`.
pub fn compute_cbp_luma_ctx_idx_inc_bin(
    bin_idx: u32,
    current_partial_cbp_luma: u8,
    neighbors: &CabacNeighborContext,
    mb_x: usize,
) -> u32 {
    // binIdx → current 8×8 position in MB coords.
    let cur_bx = (bin_idx & 1) as u8;
    let cur_by = ((bin_idx >> 1) & 1) as u8;

    // Neighbor A (left).
    let cond_a = if cur_bx > 0 {
        // Same-MB neighbor at (cur_bx - 1, cur_by): raster bin_idx =
        // cur_by * 2 + (cur_bx - 1).
        let neighbor_bin = cur_by * 2 + (cur_bx - 1);
        if (current_partial_cbp_luma >> neighbor_bin) & 1 == 0 {
            1
        } else {
            0
        }
    } else {
        // Cross-MB left: neighbor MB's 8×8 at (1, cur_by) = bin_idx
        // (cur_by * 2 + 1).
        let neighbor_bin = (cur_by * 2 + 1) as u32;
        cbp_luma_cross_cond(neighbors.neighbor_a(mb_x), neighbor_bin)
    };
    // Neighbor B (above).
    let cond_b = if cur_by > 0 {
        let neighbor_bin = (cur_by - 1) * 2 + cur_bx;
        if (current_partial_cbp_luma >> neighbor_bin) & 1 == 0 {
            1
        } else {
            0
        }
    } else {
        // Cross-MB top: neighbor MB's 8×8 at (cur_bx, 1) = bin_idx
        // (1 * 2 + cur_bx).
        let neighbor_bin = (2 + cur_bx) as u32;
        cbp_luma_cross_cond(neighbors.neighbor_b(mb_x), neighbor_bin)
    };
    cond_a + 2 * cond_b
}

/// Compute ctxIdxInc for `coded_block_pattern` chroma bin (spec
/// § 9.3.3.1.1.4, eq 9-11). `ctxIdxInc = condTermFlagA +
/// 2·condTermFlagB + (binIdx == 1 ? 4 : 0)`.
pub fn ctx_idx_inc_cbp_chroma(
    ctx: &CabacNeighborContext,
    mb_x: usize,
    bin_idx: u32,
) -> u32 {
    let cond = |n: Option<&CabacNeighborMB>| -> u32 {
        match n {
            None => 0,
            Some(mb) if mb.mb_type == MbTypeClass::IPCM => 1,
            Some(mb) if mb.mb_type.is_skip() => 0,
            Some(mb) => {
                // binIdx 0: any chroma coded at all? binIdx 1: chroma AC coded?
                if bin_idx == 0 {
                    if mb.cbp_chroma != 0 { 1 } else { 0 }
                } else {
                    if mb.cbp_chroma == 2 { 1 } else { 0 }
                }
            }
        }
    };
    let base = cond(ctx.neighbor_a(mb_x)) + 2 * cond(ctx.neighbor_b(mb_x));
    base + if bin_idx == 1 { 4 } else { 0 }
}

/// Compute ctxIdxInc for `coded_block_flag` (spec § 9.3.3.1.1.9,
/// eq 9-19). `ctxIdxInc = condTermFlagA + 2·condTermFlagB`.
///
/// `ctx_block_cat` is the block category (0..4 for 4:2:0 Phase 6C
/// scope). `block_idx_in_mb_{a,b}` are the per-category block
/// indices in the respective neighbor.
pub fn ctx_idx_inc_coded_block_flag(
    ctx: &CabacNeighborContext,
    mb_x: usize,
    ctx_block_cat: u8,
    block_idx_in_mb_a: usize,
    block_idx_in_mb_b: usize,
    current_is_intra: bool,
) -> u32 {
    let cond = |n: Option<&CabacNeighborMB>, blk: usize| -> u32 {
        match n {
            // Spec § 9.3.3.1.1.9: when the neighbour MB is not
            // available (off-frame, different slice, or out-of-
            // picture), `coded_block_flag[mbAddrN]` is inferred to
            // 1 if the current MB is intra (or I_PCM) and 0 if the
            // current MB is inter. This matches the spec's
            // "inference from zero/one" rule for unavailable
            // neighbours combined with the intra-pred-rule override.
            // A prior implementation collapsed both branches to 0,
            // which produced a ctxIdxInc mismatch on any intra MB
            // at the top/left frame boundary.
            None => if current_is_intra { 1 } else { 0 },
            Some(mb) if mb.mb_type == MbTypeClass::IPCM => 1,
            Some(mb) => ((neighbor_cbf_bitmap(mb, ctx_block_cat) >> blk) & 1) as u32,
        }
    };
    cond(ctx.neighbor_a(mb_x), block_idx_in_mb_a)
        + 2 * cond(ctx.neighbor_b(mb_x), block_idx_in_mb_b)
}

/// Returns the neighbour's effective CBF bitmap for the given
/// ctxBlockCat. Under spec § 9.3.3.1.1.9 a 4×4 neighbour block's
/// coded_block_flag is a single per-position attribute regardless
/// of whether that position lives in an Intra16x16AC residual
/// (cat = 1) or a Luma4x4 residual (cat = 2) — both categories
/// map back to the same underlying "this 4×4 position had
/// non-zero coefficients" state. Our `coded_block_flag_cat[]`
/// array has separate slots per cat, so we union cats 1 and 2 on
/// lookup to present the spec-required single-per-position view.
/// Without the union, a neighbour coded as I_4x4 (cat = 2) would
/// answer cat = 1 (Intra16x16AC) queries as zero and produce an
/// off-by-one ctxIdxInc.
#[inline]
fn neighbor_cbf_bitmap(mb: &CabacNeighborMB, ctx_block_cat: u8) -> u16 {
    match ctx_block_cat {
        1 | 2 => mb.coded_block_flag_cat[1] | mb.coded_block_flag_cat[2],
        _ => mb.coded_block_flag_cat[ctx_block_cat as usize],
    }
}

/// Compute ctxIdxInc for `significant_coeff_flag` and
/// `last_significant_coeff_flag` for 4×4 blocks (spec § 9.3.3.1.3
/// eq 9-21): `ctxIdxInc = levelListIdx` (the scan position).
///
/// For 4×4 luma / chroma AC blocks (ctxBlockCat ∈ {0, 1, 2, 4})
/// this ranges 0..14. For chroma DC (ctxBlockCat = 3) the rule is
/// eq 9-22 — use `ctx_idx_inc_sig_chroma_dc` instead.
pub fn ctx_idx_inc_sig_4x4(level_list_idx: u32) -> u32 {
    level_list_idx
}

/// Compute ctxIdxInc for chroma-DC significance flags (spec eq 9-22).
/// `ctxIdxInc = Min(levelListIdx / NumC8x8, 2)`.
///
/// For 4:2:0 (our scope), `NumC8x8 = 1` → this reduces to
/// `Min(levelListIdx, 2)`.
pub fn ctx_idx_inc_sig_chroma_dc(level_list_idx: u32) -> u32 {
    level_list_idx.min(2)
}

/// Compute ctxIdxInc for `coeff_abs_level_minus1` (spec § 9.3.3.1.3
/// eqs 9-23 and 9-24).
///
/// `bin_idx == 0`: "is |coeff| >= 2?" bin.
///   ctxIdxInc = numGt1 != 0 ? 0 : Min(4, 1 + numEq1).
/// `bin_idx > 0`: TU prefix tail.
///   ctxIdxInc = 5 + Min(4 - (cat==3 ? 1 : 0), numGt1).
pub fn ctx_idx_inc_coeff_abs_level(
    bin_idx: u32,
    ctx_block_cat: u8,
    num_decod_abs_level_eq1: u32,
    num_decod_abs_level_gt1: u32,
) -> u32 {
    if bin_idx == 0 {
        if num_decod_abs_level_gt1 != 0 {
            0
        } else {
            (1 + num_decod_abs_level_eq1).min(4)
        }
    } else {
        let cap = 4 - if ctx_block_cat == 3 { 1 } else { 0 };
        5 + num_decod_abs_level_gt1.min(cap)
    }
}

/// Compute ctxIdxInc for `transform_size_8x8_flag` (spec § 9.3.3.1.1.10,
/// eq 9-20). High-profile only. `ctxIdxInc = condTermFlagA + condTermFlagB`
/// where `condTermFlagN = 1` iff neighbor N is available AND
/// `transform_size_8x8_flag[N] == 1`, else 0.
pub fn ctx_idx_inc_transform_size_8x8_flag(
    ctx: &CabacNeighborContext,
    mb_x: usize,
) -> u32 {
    let cond = |n: Option<&CabacNeighborMB>| -> u32 {
        match n {
            Some(mb) if mb.transform_size_8x8_flag => 1,
            _ => 0,
        }
    };
    let _ = cond;
    // Use the closure-like inline lookup via neighbor_a / neighbor_b.
    let a = match ctx.neighbor_a(mb_x) {
        Some(mb) if mb.transform_size_8x8_flag => 1,
        _ => 0,
    };
    let b = match ctx.neighbor_b(mb_x) {
        Some(mb) if mb.transform_size_8x8_flag => 1,
        _ => 0,
    };
    a + b
}

// ─── Current-MB coded_block_flag tracking (Phase 6C.6d) ───────────
//
// The `coded_block_flag` ctxIdxInc derivation (§ 9.3.3.1.1.9) pulls
// `condTermFlagN` from the 4×4-block neighbor N. When N is in a
// DIFFERENT MB (cross-MB: current block on the MB's left/top edge),
// the lookup goes through `CabacNeighborContext::neighbor_a/b`. When
// N is in the SAME MB (interior block), we need the progressively-
// built state of the current MB. `CurrentMbCbf` holds that state.

/// Progressive coded_block_flag state for the current MB, built up
/// residual-block by residual-block as they emit.
#[derive(Debug, Default, Clone, Copy)]
pub struct CurrentMbCbf {
    /// One u16 bitmap per ctxBlockCat (0..=4). Layout matches
    /// `CabacNeighborMB::coded_block_flag_cat`:
    ///   cat 0: 1 slot  (luma DC, bit 0)
    ///   cat 1: 16 slots (luma AC per 4×4 block)
    ///   cat 2: 16 slots (luma 4×4 for I_4x4 / inter)
    ///   cat 3: 2 slots (chroma DC per plane)
    ///   cat 4: 8 slots (chroma AC: 4 per plane × 2 planes)
    pub cat: [u16; 5],
}

impl CurrentMbCbf {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn get(&self, cat: u8, block_idx: usize) -> bool {
        (self.cat[cat as usize] >> block_idx) & 1 != 0
    }

    #[inline]
    pub fn set(&mut self, cat: u8, block_idx: usize, value: bool) {
        if value {
            self.cat[cat as usize] |= 1 << block_idx;
        }
    }

    /// Freeze into the `coded_block_flag_cat` layout consumed by
    /// `CabacNeighborMB`.
    #[inline]
    pub fn to_neighbor_cbf(&self) -> [u16; 5] {
        self.cat
    }
}

/// Progressive `abs_mvd_comp` state for the current MB, built up as
/// motion partitions emit. Used by the MVD bin-0 ctxIdxInc derivation
/// when the neighbor block falls INSIDE the current MB (partition 1
/// of P_16x8, right partition of P_8x16, interior sub-MB partitions
/// of P_8x8). Layout mirrors `CabacNeighborMB::abs_mvd_comp`:
/// `[component][block_idx_in_mb]` with block_idx in 0..=15 using the
/// BLOCK_INDEX_TO_POS ordering.
#[derive(Debug, Default, Clone, Copy)]
pub struct CurrentMbMvdAbs {
    pub comp: [[i16; 16]; 2],
}

impl CurrentMbMvdAbs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fill a rectangular 4×4-block region within the current MB with
    /// this partition's absolute MVD values. The region is (bx0..bx0+w)
    /// × (by0..by0+h) in 4×4-block coords.
    pub fn fill_region(&mut self, bx0: u8, by0: u8, w: u8, h: u8, abs_mvd_x: i16, abs_mvd_y: i16) {
        for dy in 0..h {
            for dx in 0..w {
                let idx = block_pos_to_luma_idx(bx0 + dx, by0 + dy);
                self.comp[0][idx] = abs_mvd_x;
                self.comp[1][idx] = abs_mvd_y;
            }
        }
    }

    #[inline]
    pub fn get(&self, component: u8, block_idx: usize) -> i16 {
        self.comp[component as usize][block_idx]
    }

    #[inline]
    pub fn to_neighbor(&self) -> [[i16; 16]; 2] {
        self.comp
    }
}

/// Compute ctxIdxInc for mvd bin 0 (spec § 9.3.3.1.1.7 eq 9-15) with
/// proper same-MB / cross-MB neighbor selection. `cur_bx`, `cur_by`
/// are the partition's top-left 4×4-block coordinates within the
/// current MB.
pub fn compute_mvd_ctx_idx_inc_bin0(
    current_mb: &CurrentMbMvdAbs,
    neighbors: &CabacNeighborContext,
    mb_x: usize,
    cur_bx: u8,
    cur_by: u8,
    component: u8,
) -> u32 {
    let abs_a: u32 = if cur_bx > 0 {
        current_mb
            .get(component, block_pos_to_luma_idx(cur_bx - 1, cur_by))
            .unsigned_abs() as u32
    } else {
        match neighbors.neighbor_a(mb_x) {
            Some(mb) => mb.abs_mvd_comp[component as usize]
                [block_pos_to_luma_idx(3, cur_by)]
                .unsigned_abs() as u32,
            None => 0,
        }
    };
    let abs_b: u32 = if cur_by > 0 {
        current_mb
            .get(component, block_pos_to_luma_idx(cur_bx, cur_by - 1))
            .unsigned_abs() as u32
    } else {
        match neighbors.neighbor_b(mb_x) {
            Some(mb) => mb.abs_mvd_comp[component as usize]
                [block_pos_to_luma_idx(cur_bx, 3)]
                .unsigned_abs() as u32,
            None => 0,
        }
    };
    let sum = abs_a + abs_b;
    if sum < 3 {
        0
    } else if sum > 32 {
        2
    } else {
        1
    }
}

/// Map a 4×4-luma-block position (bx, by) in MB coords (each in 0..=3)
/// to the H.264 BlockIndex k (0..=15) — the raster-in-8×8, 8×8-raster
/// ordering from `BLOCK_INDEX_TO_POS`.
#[inline]
pub fn block_pos_to_luma_idx(bx: u8, by: u8) -> usize {
    debug_assert!(bx < 4 && by < 4);
    let bx = bx as usize;
    let by = by as usize;
    // 8x8-block slot (raster of 4 8x8s) × 4 + 4x4-slot-within-8x8.
    4 * (2 * (by / 2) + (bx / 2)) + (2 * (by % 2) + (bx % 2))
}

/// Map a chroma 4×4-block position (bx, by) in MB coords (each 0..=1)
/// for plane `i_cb_cr` (0 = Cb, 1 = Cr) to the ctxBlockCat=4 slot
/// (0..=7). Layout: bits 0..3 = Cb's four blocks, bits 4..7 = Cr's.
#[inline]
pub fn block_pos_to_chroma_ac_idx(i_cb_cr: u8, bx: u8, by: u8) -> usize {
    debug_assert!(bx < 2 && by < 2 && i_cb_cr < 2);
    (i_cb_cr as usize) * 4 + (by as usize) * 2 + (bx as usize)
}

/// Compute ctxIdxInc for coded_block_flag at ctxBlockCat 0
/// (Intra16x16DCLevel). Single block per MB — neighbors are always
/// cross-MB. Matches existing `ctx_idx_inc_coded_block_flag` called
/// with (cat=0, block_a=0, block_b=0).
pub fn compute_cbf_ctx_idx_inc_luma_dc(
    neighbors: &CabacNeighborContext,
    mb_x: usize,
) -> u32 {
    // Luma16x16 DC is only emitted in I_16x16 MBs, which are always
    // intra → pass true.
    ctx_idx_inc_coded_block_flag(neighbors, mb_x, 0, 0, 0, true)
}

/// Compute ctxIdxInc for coded_block_flag at ctxBlockCat 1
/// (Intra16x16ACLevel). The neighbor is same-MB when the current
/// block is interior (bx > 0 for A, by > 0 for B) and cross-MB
/// otherwise.
pub fn compute_cbf_ctx_idx_inc_luma_ac(
    current_mb: &CurrentMbCbf,
    neighbors: &CabacNeighborContext,
    mb_x: usize,
    bx: u8,
    by: u8,
    current_is_intra: bool,
) -> u32 {
    const CAT: u8 = 1;
    let cond_term_a = if bx > 0 {
        current_mb.get(CAT, block_pos_to_luma_idx(bx - 1, by)) as u32
    } else {
        cbf_cross_mb_condterm(
            neighbors.neighbor_a(mb_x),
            CAT,
            block_pos_to_luma_idx(3, by),
            current_is_intra,
        )
    };
    let cond_term_b = if by > 0 {
        current_mb.get(CAT, block_pos_to_luma_idx(bx, by - 1)) as u32
    } else {
        cbf_cross_mb_condterm(
            neighbors.neighbor_b(mb_x),
            CAT,
            block_pos_to_luma_idx(bx, 3),
            current_is_intra,
        )
    };
    cond_term_a + 2 * cond_term_b
}

/// Compute ctxIdxInc for coded_block_flag at ctxBlockCat 2
/// (LumaLevel4x4 — used by I_4x4 and P/B inter). Identical mechanics
/// to `compute_cbf_ctx_idx_inc_luma_ac` but targets ctxBlockCat 2.
pub fn compute_cbf_ctx_idx_inc_luma_4x4(
    current_mb: &CurrentMbCbf,
    neighbors: &CabacNeighborContext,
    mb_x: usize,
    bx: u8,
    by: u8,
    current_is_intra: bool,
) -> u32 {
    const CAT: u8 = 2;
    let cond_term_a = if bx > 0 {
        current_mb.get(CAT, block_pos_to_luma_idx(bx - 1, by)) as u32
    } else {
        cbf_cross_mb_condterm(
            neighbors.neighbor_a(mb_x),
            CAT,
            block_pos_to_luma_idx(3, by),
            current_is_intra,
        )
    };
    let cond_term_b = if by > 0 {
        current_mb.get(CAT, block_pos_to_luma_idx(bx, by - 1)) as u32
    } else {
        cbf_cross_mb_condterm(
            neighbors.neighbor_b(mb_x),
            CAT,
            block_pos_to_luma_idx(bx, 3),
            current_is_intra,
        )
    };
    cond_term_a + 2 * cond_term_b
}

/// Compute ctxIdxInc for coded_block_flag at ctxBlockCat 3 (ChromaDCLevel).
/// One block per chroma plane per MB — neighbors always cross-MB.
pub fn compute_cbf_ctx_idx_inc_chroma_dc(
    neighbors: &CabacNeighborContext,
    mb_x: usize,
    i_cb_cr: u8,
    current_is_intra: bool,
) -> u32 {
    debug_assert!(i_cb_cr < 2);
    ctx_idx_inc_coded_block_flag(
        neighbors,
        mb_x,
        3,
        i_cb_cr as usize,
        i_cb_cr as usize,
        current_is_intra,
    )
}

/// Compute ctxIdxInc for coded_block_flag at ctxBlockCat 4 (ChromaACLevel).
/// Chroma 8×8 plane hosts a 2×2 grid of 4×4 blocks. Neighbors are
/// same-MB for interior blocks (bx > 0 or by > 0) and cross-MB
/// otherwise.
pub fn compute_cbf_ctx_idx_inc_chroma_ac(
    current_mb: &CurrentMbCbf,
    neighbors: &CabacNeighborContext,
    mb_x: usize,
    i_cb_cr: u8,
    bx: u8,
    by: u8,
    current_is_intra: bool,
) -> u32 {
    debug_assert!(i_cb_cr < 2 && bx < 2 && by < 2);
    const CAT: u8 = 4;
    let cond_term_a = if bx > 0 {
        current_mb.get(CAT, block_pos_to_chroma_ac_idx(i_cb_cr, bx - 1, by)) as u32
    } else {
        cbf_cross_mb_condterm(
            neighbors.neighbor_a(mb_x),
            CAT,
            block_pos_to_chroma_ac_idx(i_cb_cr, 1, by),
            current_is_intra,
        )
    };
    let cond_term_b = if by > 0 {
        current_mb.get(CAT, block_pos_to_chroma_ac_idx(i_cb_cr, bx, by - 1)) as u32
    } else {
        cbf_cross_mb_condterm(
            neighbors.neighbor_b(mb_x),
            CAT,
            block_pos_to_chroma_ac_idx(i_cb_cr, bx, 1),
            current_is_intra,
        )
    };
    cond_term_a + 2 * cond_term_b
}

#[inline]
fn cbf_cross_mb_condterm(
    n: Option<&CabacNeighborMB>,
    ctx_block_cat: u8,
    blk: usize,
    current_is_intra: bool,
) -> u32 {
    match n {
        None => if current_is_intra { 1 } else { 0 },
        Some(mb) if mb.mb_type == MbTypeClass::IPCM => 1,
        Some(mb) => ((neighbor_cbf_bitmap(mb, ctx_block_cat) >> blk) & 1) as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> CabacNeighborContext {
        CabacNeighborContext::new(4, CabacInitSlot::ISI)
    }

    fn p_inter_mb() -> CabacNeighborMB {
        CabacNeighborMB {
            mb_type: MbTypeClass::PInter,
            ..Default::default()
        }
    }

    fn p_skip_mb() -> CabacNeighborMB {
        CabacNeighborMB {
            mb_type: MbTypeClass::PSkip,
            mb_skip_flag: true,
            ..Default::default()
        }
    }

    fn intra_nxn_mb() -> CabacNeighborMB {
        CabacNeighborMB {
            mb_type: MbTypeClass::INxN,
            ..Default::default()
        }
    }

    #[test]
    fn mb_skip_flag_no_neighbors_is_zero() {
        let ctx = make_ctx();
        assert_eq!(ctx_idx_inc_mb_skip_flag(&ctx, 0), 0);
    }

    #[test]
    fn mb_skip_flag_with_both_non_skip_neighbors_is_two() {
        let mut ctx = make_ctx();
        ctx.top_row[0] = Some(p_inter_mb());
        ctx.left = Some(p_inter_mb());
        assert_eq!(ctx_idx_inc_mb_skip_flag(&ctx, 0), 2);
    }

    #[test]
    fn mb_skip_flag_skip_neighbor_contributes_zero() {
        let mut ctx = make_ctx();
        ctx.top_row[0] = Some(p_skip_mb());
        ctx.left = Some(p_inter_mb());
        assert_eq!(ctx_idx_inc_mb_skip_flag(&ctx, 0), 1);
    }

    #[test]
    fn mb_type_bin0_i_slice_nxn_contributes_zero() {
        // I-slice ctxIdxOffset=3. Neighbor is I_NxN → contributes 0.
        let mut ctx = make_ctx();
        ctx.left = Some(intra_nxn_mb());
        ctx.top_row[0] = Some(CabacNeighborMB {
            mb_type: MbTypeClass::I16x16,
            ..Default::default()
        });
        // Left contributes 0 (I_NxN), top contributes 1 (I_16x16).
        assert_eq!(ctx_idx_inc_mb_type_bin0(&ctx, 0, 3), 1);
    }

    #[test]
    fn mb_type_bin0_p_slice_uses_fixed_zero() {
        // P-slice mb_type prefix bin 0 at ctxIdxOffset=14 uses FIXED
        // ctxIdxInc=0 per spec Table 9-39 (no neighbor derivation for
        // this offset). Task #21 root-cause fix (2026-04-23): an
        // earlier neighbor-based rule caused desync at MBs where
        // left neighbor was intra-in-P.
        let mut ctx = make_ctx();
        ctx.left = Some(p_inter_mb());
        ctx.top_row[0] = Some(intra_nxn_mb());
        assert_eq!(ctx_idx_inc_mb_type_bin0(&ctx, 0, 14), 0);
        // Also test with both neighbors intra — still 0.
        ctx.left = Some(intra_nxn_mb());
        ctx.top_row[0] = Some(intra_nxn_mb());
        assert_eq!(ctx_idx_inc_mb_type_bin0(&ctx, 0, 14), 0);
    }

    #[test]
    fn prior_bin_table_9_41_lookups() {
        assert_eq!(ctx_idx_inc_prior_bin(3, 4, &[0, 0, 0, 1]), Some(5));
        assert_eq!(ctx_idx_inc_prior_bin(3, 4, &[0, 0, 0, 0]), Some(6));
        assert_eq!(ctx_idx_inc_prior_bin(14, 2, &[0, 0]), Some(2));
        assert_eq!(ctx_idx_inc_prior_bin(14, 2, &[0, 1]), Some(3));
        assert_eq!(ctx_idx_inc_prior_bin(99, 0, &[]), None);
    }

    #[test]
    fn mvd_bin0_range_test() {
        let mut ctx = make_ctx();
        let mut a = CabacNeighborMB::default();
        a.abs_mvd_comp[0][0] = 1;
        let mut b = CabacNeighborMB::default();
        b.abs_mvd_comp[0][0] = 1;
        ctx.left = Some(a);
        ctx.top_row[0] = Some(b);
        // sum = 2 < 3 → ctxIdxInc = 0.
        assert_eq!(ctx_idx_inc_mvd_bin0(&ctx, 0, 0, 0, 0), 0);

        a.abs_mvd_comp[0][0] = 10;
        b.abs_mvd_comp[0][0] = 10;
        ctx.left = Some(a);
        ctx.top_row[0] = Some(b);
        // sum = 20 in [3, 32] → ctxIdxInc = 1.
        assert_eq!(ctx_idx_inc_mvd_bin0(&ctx, 0, 0, 0, 0), 1);

        a.abs_mvd_comp[0][0] = 40;
        ctx.left = Some(a);
        // sum = 50 > 32 → ctxIdxInc = 2.
        assert_eq!(ctx_idx_inc_mvd_bin0(&ctx, 0, 0, 0, 0), 2);
    }

    #[test]
    fn ref_idx_bin0_zero_ref_contributes_zero() {
        let mut ctx = make_ctx();
        let mut mb = p_inter_mb();
        mb.ref_idx_l0[0] = 0; // zero ref → contributes 0
        ctx.left = Some(mb);
        ctx.top_row[0] = Some(mb);
        assert_eq!(ctx_idx_inc_ref_idx_bin0(&ctx, 0, 0, 0), 0);
    }

    #[test]
    fn ref_idx_bin0_nonzero_ref_contributes() {
        let mut ctx = make_ctx();
        let mut a = p_inter_mb();
        a.ref_idx_l0[0] = 1;
        let b = p_inter_mb(); // ref_idx_l0 default = 0
        ctx.left = Some(a);
        ctx.top_row[0] = Some(b);
        // A contributes 1, B contributes 0 → ctxIdxInc = 1 + 2*0 = 1.
        assert_eq!(ctx_idx_inc_ref_idx_bin0(&ctx, 0, 0, 0), 1);
    }

    #[test]
    fn mb_qp_delta_prev_zero_or_unavail_is_zero() {
        let ctx = make_ctx();
        assert_eq!(ctx_idx_inc_mb_qp_delta_bin0(&ctx), 0);
    }

    #[test]
    fn mb_qp_delta_prev_nonzero_is_one() {
        let mut ctx = make_ctx();
        let mut prev = intra_nxn_mb();
        prev.mb_qp_delta = 3;
        ctx.prev_mb = Some(prev);
        assert_eq!(ctx_idx_inc_mb_qp_delta_bin0(&ctx), 1);
    }

    #[test]
    fn sig_4x4_returns_level_list_idx() {
        assert_eq!(ctx_idx_inc_sig_4x4(0), 0);
        assert_eq!(ctx_idx_inc_sig_4x4(7), 7);
        assert_eq!(ctx_idx_inc_sig_4x4(14), 14);
    }

    #[test]
    fn sig_chroma_dc_clamps_at_2() {
        assert_eq!(ctx_idx_inc_sig_chroma_dc(0), 0);
        assert_eq!(ctx_idx_inc_sig_chroma_dc(1), 1);
        assert_eq!(ctx_idx_inc_sig_chroma_dc(2), 2);
        assert_eq!(ctx_idx_inc_sig_chroma_dc(5), 2);
    }

    #[test]
    fn coeff_abs_level_bin0_no_prior_gt1() {
        // numGt1 = 0, numEq1 = 0 → ctxIdxInc = min(4, 1 + 0) = 1.
        assert_eq!(ctx_idx_inc_coeff_abs_level(0, 2, 0, 0), 1);
        // numGt1 = 0, numEq1 = 3 → ctxIdxInc = min(4, 4) = 4.
        assert_eq!(ctx_idx_inc_coeff_abs_level(0, 2, 3, 0), 4);
        // numGt1 = 0, numEq1 = 10 → min(4, 11) = 4.
        assert_eq!(ctx_idx_inc_coeff_abs_level(0, 2, 10, 0), 4);
    }

    #[test]
    fn coeff_abs_level_bin0_with_prior_gt1_is_zero() {
        assert_eq!(ctx_idx_inc_coeff_abs_level(0, 2, 5, 1), 0);
    }

    #[test]
    fn coeff_abs_level_subsequent_bins_chroma_dc_cap() {
        // cat=3 (chroma DC) → cap = 4 - 1 = 3.
        assert_eq!(ctx_idx_inc_coeff_abs_level(1, 3, 0, 0), 5);
        assert_eq!(ctx_idx_inc_coeff_abs_level(1, 3, 0, 10), 5 + 3); // capped
        // Other cats → cap = 4.
        assert_eq!(ctx_idx_inc_coeff_abs_level(1, 2, 0, 10), 5 + 4);
    }

    #[test]
    fn commit_and_reset_cycle() {
        let mut ctx = make_ctx();
        ctx.commit(0, intra_nxn_mb());
        assert!(ctx.neighbor_a(1).is_some());
        assert!(ctx.neighbor_b(0).is_some());
        ctx.reset();
        assert!(ctx.neighbor_a(0).is_none());
        assert!(ctx.neighbor_b(0).is_none());
        assert!(ctx.prev_mb().is_none());
    }

    #[test]
    fn new_row_clears_left_only() {
        let mut ctx = make_ctx();
        ctx.commit(0, intra_nxn_mb());
        ctx.new_row();
        assert!(ctx.neighbor_a(0).is_none());
        // Top-row entry survives.
        assert!(ctx.neighbor_b(0).is_some());
    }

    // ─── Phase 6C.6d current-MB CBF tracking tests ──────────────

    #[test]
    fn block_pos_to_luma_idx_matches_block_index_to_pos() {
        use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;
        for (expected_idx, &(bx, by)) in BLOCK_INDEX_TO_POS.iter().enumerate() {
            let idx = block_pos_to_luma_idx(bx, by);
            assert_eq!(
                idx, expected_idx,
                "block_pos_to_luma_idx({bx},{by}) should be {expected_idx}, got {idx}"
            );
        }
    }

    #[test]
    fn block_pos_to_chroma_ac_idx_layout() {
        // Cb plane: 0..3; Cr plane: 4..7.
        assert_eq!(block_pos_to_chroma_ac_idx(0, 0, 0), 0);
        assert_eq!(block_pos_to_chroma_ac_idx(0, 1, 0), 1);
        assert_eq!(block_pos_to_chroma_ac_idx(0, 0, 1), 2);
        assert_eq!(block_pos_to_chroma_ac_idx(0, 1, 1), 3);
        assert_eq!(block_pos_to_chroma_ac_idx(1, 0, 0), 4);
        assert_eq!(block_pos_to_chroma_ac_idx(1, 1, 1), 7);
    }

    #[test]
    fn current_mb_cbf_set_get_roundtrip() {
        let mut c = CurrentMbCbf::new();
        assert!(!c.get(1, 5));
        c.set(1, 5, true);
        assert!(c.get(1, 5));
        assert!(!c.get(1, 4));
        assert!(!c.get(2, 5));
        c.set(1, 5, false); // false shouldn't clear (set-only contract)
        assert!(c.get(1, 5));
    }

    #[test]
    fn compute_cbf_ctx_idx_inc_luma_dc_cross_mb_only() {
        // Luma16x16 DC is always intra. Spec § 9.3.3.1.1.9: unavailable
        // neighbour contributes 1 for intra → fresh MB (0, 0) = 1+2 = 3.
        let mut ctx = make_ctx();
        assert_eq!(compute_cbf_ctx_idx_inc_luma_dc(&ctx, 0), 3);
        let mut nb = CabacNeighborMB::default();
        nb.mb_type = MbTypeClass::I16x16;
        nb.coded_block_flag_cat[0] = 0b1; // DC coded
        ctx.commit(0, nb);
        // mb_x=1: A = left MB (real, DC coded → 1), B unavailable intra → 2.
        assert_eq!(compute_cbf_ctx_idx_inc_luma_dc(&ctx, 1), 3);
    }

    #[test]
    fn compute_cbf_ctx_idx_inc_luma_ac_interior_uses_current_mb() {
        // Interior block at (1, 0): A = same-MB (0, 0), B = top-MB (off).
        // With intra=true: A (real cbf=1) → 1, B (unavailable intra) → 2.
        let ctx = make_ctx();
        let mut cur = CurrentMbCbf::new();
        let idx_00 = block_pos_to_luma_idx(0, 0);
        cur.set(1, idx_00, true);
        let inc = compute_cbf_ctx_idx_inc_luma_ac(&cur, &ctx, 0, 1, 0, true);
        assert_eq!(inc, 3, "condTermA=1 + 2*condTermB(intra unavail)=2 = 3");
        // With intra=false, unavailable contributes 0 → A=1, B=0 → inc=1.
        let inc = compute_cbf_ctx_idx_inc_luma_ac(&cur, &ctx, 0, 1, 0, false);
        assert_eq!(inc, 1, "condTermA=1 + 2*condTermB(inter unavail)=0 = 1");
    }

    #[test]
    fn compute_cbf_ctx_idx_inc_luma_ac_edge_uses_cross_mb() {
        // Edge block at (0, 0) of MB (0, 0): both A and B off-frame.
        let mut ctx = make_ctx();
        let cur = CurrentMbCbf::new();
        // intra=true → both unavailable → 1 + 2 = 3.
        assert_eq!(compute_cbf_ctx_idx_inc_luma_ac(&cur, &ctx, 0, 0, 0, true), 3);
        // intra=false → both unavailable → 0 + 0 = 0.
        assert_eq!(compute_cbf_ctx_idx_inc_luma_ac(&cur, &ctx, 0, 0, 0, false), 0);
        let mut nb = CabacNeighborMB::default();
        nb.mb_type = MbTypeClass::I16x16;
        nb.coded_block_flag_cat[1] = 1 << 5; // (3, 0) → idx 5
        ctx.commit(0, nb);
        // mb_x=1, block (0,0): A = real left MB cbf=1 → 1, B unavailable.
        // intra=true → B=2 → inc = 1+2 = 3.
        let inc = compute_cbf_ctx_idx_inc_luma_ac(&cur, &ctx, 1, 0, 0, true);
        assert_eq!(inc, 3);
        // intra=false → B=0 → inc = 1.
        let inc = compute_cbf_ctx_idx_inc_luma_ac(&cur, &ctx, 1, 0, 0, false);
        assert_eq!(inc, 1);
    }

    #[test]
    fn compute_cbf_ctx_idx_inc_chroma_ac_interior() {
        let ctx = make_ctx();
        let mut cur = CurrentMbCbf::new();
        cur.set(4, block_pos_to_chroma_ac_idx(0, 0, 0), true);
        // At (1, 0) in Cb: A = same-MB (0, 0) cbf=1 → 1. B = top-MB off.
        // intra=true → B=2 → inc=3.
        let inc = compute_cbf_ctx_idx_inc_chroma_ac(&cur, &ctx, 0, 0, 1, 0, true);
        assert_eq!(inc, 3);
        // intra=false → B=0 → inc=1.
        let inc = compute_cbf_ctx_idx_inc_chroma_ac(&cur, &ctx, 0, 0, 1, 0, false);
        assert_eq!(inc, 1);
    }
}
