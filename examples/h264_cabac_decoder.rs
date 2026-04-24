// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Minimal spec-direct CABAC decoder for encoder debugging.
//!
//! Implements H.264 § 9.3.3.2 DecodeDecision / DecodeBypass /
//! DecodeTerminate exactly as written in the spec. Uses the SAME
//! `CabacContext` tables as our encoder (initialize_contexts, RANGE_TAB_LPS,
//! TRANS_IDX_LPS, TRANS_IDX_MPS) so any encoder/decoder disagreement is
//! a spec-interpretation bug, not a table mismatch.
//!
//! Usage:
//! ```
//! h264_cabac_decoder <slice.h264>
//! ```
//! Decodes the first P-slice's I_16x16 MBs bin-by-bin (mb_type, chroma
//! pred, qp_delta, coded_block_flag, end_of_slice). Prints each bin's
//! value + ctxIdx + post-decode state so it can be diff'd against the
//! encoder's emit trace.
//!
//! Scope: only decodes the minimum needed to cross-check flat I_16x16
//! CABAC. No residuals, no I_4x4.

use std::env;
use std::fs;

use phasm_core::codec::h264::bitstream::{parse_nal_units_annexb, remove_emulation_prevention_with_map};
use phasm_core::codec::h264::cabac::{
    context::{initialize_contexts, CabacContext, CabacInitSlot},
    neighbor::{
        block_pos_to_luma_idx, compute_cbf_ctx_idx_inc_chroma_ac,
        compute_cbf_ctx_idx_inc_chroma_dc, compute_cbf_ctx_idx_inc_luma_4x4,
        compute_cbf_ctx_idx_inc_luma_ac, compute_cbf_ctx_idx_inc_luma_dc,
        compute_cbp_luma_ctx_idx_inc_bin, ctx_idx_inc_cbp_chroma,
        ctx_idx_inc_intra_chroma_pred_mode_bin0, ctx_idx_inc_mb_skip_flag,
        CabacNeighborContext, CabacNeighborMB, CurrentMbCbf, MbTypeClass,
    },
    tables::RANGE_TAB_LPS,
};
use phasm_core::codec::h264::transform::{
    dequant_4x4, inverse_16x16_dc_hadamard, inverse_4x4_integer,
};
use phasm_core::codec::h264::slice::parse_slice_header;
use phasm_core::codec::h264::sps::{parse_pps, parse_sps};
use phasm_core::codec::h264::NalType;

/// Spec-direct CABAC decoding engine (§ 9.3.3.2).
struct CabacDecoder<'a> {
    /// Concatenated RBSP bytes starting at slice_data byte boundary.
    data: &'a [u8],
    /// Current bit position within `data` (bit 0 = MSB of byte 0).
    bit_pos: usize,
    /// Spec `codIRange`.
    range: u32,
    /// Spec `codIOffset`.
    offset: u32,
    /// Debug log of every decoded bin.
    trace: Vec<String>,
}

impl<'a> CabacDecoder<'a> {
    /// § 9.3.3.2.1: InitializeDecodingEngine. Reads 9 initial bits.
    fn new(data: &'a [u8], start_bit: usize) -> Self {
        let mut dec = Self {
            data,
            bit_pos: start_bit,
            range: 510,
            offset: 0,
            trace: Vec::new(),
        };
        for _ in 0..9 {
            dec.offset = (dec.offset << 1) | dec.read_bit() as u32;
        }
        dec
    }

    #[inline]
    fn read_bit(&mut self) -> u8 {
        if self.bit_pos / 8 >= self.data.len() {
            return 0;
        }
        let byte = self.data[self.bit_pos / 8];
        let bit = (byte >> (7 - (self.bit_pos % 8))) & 1;
        self.bit_pos += 1;
        bit
    }

    #[inline]
    fn renorm(&mut self) {
        while self.range < 256 {
            self.range <<= 1;
            self.offset = (self.offset << 1) | self.read_bit() as u32;
        }
    }

    /// § 9.3.3.2.2: DecodeDecision.
    fn decode_decision(&mut self, ctx: &mut CabacContext, ctx_idx: u32, label: &str) -> u8 {
        let p_state = ctx.p_state_idx() as usize;
        let val_mps = ctx.val_mps();
        let q_idx = ((self.range >> 6) & 3) as usize;
        let range_lps = RANGE_TAB_LPS[p_state][q_idx] as u32;

        let pre_range = self.range;
        let pre_offset = self.offset;

        self.range -= range_lps;

        let bin = if self.offset >= self.range {
            self.offset -= self.range;
            self.range = range_lps;
            1 - val_mps
        } else {
            val_mps
        };

        // State transition.
        if bin != val_mps {
            ctx.update_lps();
        } else {
            ctx.update_mps();
        }

        // RenormD.
        self.renorm();

        self.trace.push(format!(
            "DEC {label}: ctx={ctx_idx} pre_range=0x{pre_range:x} pre_offset=0x{pre_offset:x} \
             p_state_pre={p_state} val_mps_pre={val_mps} bin={bin} \
             post_range=0x{:x} post_offset=0x{:x} post_state={} post_mps={}",
            self.range, self.offset, ctx.p_state_idx(), ctx.val_mps()
        ));
        bin
    }

    /// § 9.3.3.2.3: DecodeBypass.
    #[allow(dead_code)]
    fn decode_bypass(&mut self) -> u8 {
        self.offset = (self.offset << 1) | self.read_bit() as u32;
        if self.offset >= self.range {
            self.offset -= self.range;
            1
        } else {
            0
        }
    }

    /// § 9.3.3.2.3: DecodeBypass — labeled trace variant for residuals.
    fn decode_bypass_lbl(&mut self, label: &str) -> u8 {
        let pre_offset = self.offset;
        let bin = self.decode_bypass();
        self.trace.push(format!(
            "DEC {label}: BYPASS pre_offset=0x{pre_offset:x} bin={bin} post_offset=0x{:x}",
            self.offset
        ));
        bin
    }

    /// § 9.3.3.2.4: DecodeTerminate.
    fn decode_terminate(&mut self, label: &str) -> u8 {
        let pre_range = self.range;
        let pre_offset = self.offset;
        self.range -= 2;
        let bin = if self.offset >= self.range {
            1 // end of slice
        } else {
            self.renorm();
            0
        };
        self.trace.push(format!(
            "DEC {label}: TERMINATE pre_range=0x{pre_range:x} pre_offset=0x{pre_offset:x} \
             bin={bin} post_range=0x{:x} post_offset=0x{:x}",
            self.range, self.offset
        ));
        bin
    }
}

// ────────────────────────────────────────────────────────────────────
// Residual block decoder (§ 9.3.3.1.3 + § 9.3.3.2 + § 9.3.4.1.1).
// Decodes a single 4x4 (or chroma DC) block via:
//   1. coded_block_flag
//   2. significance map (forward scan: sig + last per position)
//   3. abs_level + sign (reverse scan)

const CODED_BLOCK_FLAG_LOW: u32 = 85;
const SIGNIFICANT_COEFF_FLAG_FRAME_LOW: u32 = 105;
const LAST_SIGNIFICANT_COEFF_FLAG_FRAME_LOW: u32 = 166;
const COEFF_ABS_LEVEL_MINUS1_LOW: u32 = 227;

const CTX_BLOCK_CAT_OFFSET: [[u32; 5]; 4] = [
    [0, 4, 8, 12, 16],
    [0, 15, 29, 44, 47],
    [0, 15, 29, 44, 47],
    [0, 10, 20, 30, 39],
];

/// Decode a residual block per spec § 9.3.3.1.3. Returns the
/// reconstructed coefficients (in scan order, range start_idx..=end_idx).
/// `cbf_ctx_idx_inc` is computed by the caller from neighbor context.
fn decode_residual_block(
    dec: &mut CabacDecoder,
    contexts: &mut [CabacContext; 1024],
    start_idx: usize,
    end_idx: usize,
    ctx_block_cat: u8,
    cbf_ctx_idx_inc: u32,
    label: &str,
) -> Vec<i32> {
    let max_num_coeff = end_idx + 1;
    let mut coeffs = vec![0i32; max_num_coeff];

    // 1. coded_block_flag.
    let cbf_ctx_idx = CODED_BLOCK_FLAG_LOW
        + CTX_BLOCK_CAT_OFFSET[0][ctx_block_cat as usize]
        + cbf_ctx_idx_inc;
    let cbf = dec.decode_decision(
        &mut contexts[cbf_ctx_idx as usize],
        cbf_ctx_idx,
        &format!("{label} cbf"),
    );
    if cbf == 0 {
        dec.trace.push(format!("{label} → cbf=0 (no residuals)"));
        return coeffs;
    }

    // 2. Significance map (forward scan).
    let sig_offset =
        SIGNIFICANT_COEFF_FLAG_FRAME_LOW + CTX_BLOCK_CAT_OFFSET[1][ctx_block_cat as usize];
    let last_offset =
        LAST_SIGNIFICANT_COEFF_FLAG_FRAME_LOW + CTX_BLOCK_CAT_OFFSET[2][ctx_block_cat as usize];

    let mut sig_positions: Vec<usize> = Vec::new();
    let mut found_last = false;
    let mut i = start_idx;
    while i < end_idx {
        let level_list_idx = (i - start_idx) as u32;
        let sig_inc = if ctx_block_cat == 3 {
            level_list_idx.min(2)
        } else {
            level_list_idx
        };
        let sig_ctx = sig_offset + sig_inc;
        let sig = dec.decode_decision(
            &mut contexts[sig_ctx as usize],
            sig_ctx,
            &format!("{label} sig[{level_list_idx}]"),
        );
        if sig == 1 {
            sig_positions.push(i);
            let last_inc = sig_inc;
            let last_ctx = last_offset + last_inc;
            let last = dec.decode_decision(
                &mut contexts[last_ctx as usize],
                last_ctx,
                &format!("{label} last[{level_list_idx}]"),
            );
            if last == 1 {
                found_last = true;
                break;
            }
        }
        i += 1;
    }
    if !found_last {
        // The implicit last position (end_idx) is significant if reached.
        sig_positions.push(end_idx);
    }

    dec.trace.push(format!(
        "{label} → sig_positions = {:?}",
        sig_positions
    ));

    // 3. Reverse-scan abs_level + sign.
    let abs_offset =
        COEFF_ABS_LEVEL_MINUS1_LOW + CTX_BLOCK_CAT_OFFSET[3][ctx_block_cat as usize];
    let mut num_eq1 = 0u32;
    let mut num_gt1 = 0u32;

    for &pos in sig_positions.iter().rev() {
        // bin_idx 0: "is |coeff| >= 2?"
        let inc0 = if num_gt1 != 0 {
            0
        } else {
            (1 + num_eq1).min(4)
        };
        let geq2_ctx = abs_offset + inc0;
        let geq2 = dec.decode_decision(
            &mut contexts[geq2_ctx as usize],
            geq2_ctx,
            &format!("{label} abs[{pos}] geq2"),
        );
        let mut abs_minus1 = 0u32;
        if geq2 == 1 {
            // TU prefix: read 1 bins until we hit a 0 or hit cap=14.
            let inc_tail = 5 + num_gt1.min(4 - if ctx_block_cat == 3 { 1 } else { 0 });
            let tail_ctx = abs_offset + inc_tail;
            let mut prefix = 1u32;
            loop {
                let b = dec.decode_decision(
                    &mut contexts[tail_ctx as usize],
                    tail_ctx,
                    &format!("{label} abs[{pos}] tail[{prefix}]"),
                );
                if b == 0 {
                    break;
                }
                prefix += 1;
                if prefix >= 14 {
                    // EG0 suffix follows via bypass.
                    let mut suf_k = 0u32;
                    let mut suf_acc = 0u32;
                    // EG0 suffix: read 1s until 0, then k more bypass bits.
                    loop {
                        let pb = dec.decode_bypass_lbl(&format!(
                            "{label} abs[{pos}] eg0 prefix"
                        ));
                        if pb == 0 {
                            break;
                        }
                        suf_k += 1;
                    }
                    for _ in 0..suf_k {
                        let pb = dec.decode_bypass_lbl(&format!(
                            "{label} abs[{pos}] eg0 mantissa"
                        ));
                        suf_acc = (suf_acc << 1) | pb as u32;
                    }
                    let suf_value = (1u32 << suf_k) + suf_acc - 1;
                    prefix += suf_value;
                    break;
                }
            }
            abs_minus1 = prefix;
        }

        let sign = dec.decode_bypass_lbl(&format!("{label} abs[{pos}] sign"));
        let abs_val = (abs_minus1 + 1) as i32;
        coeffs[pos] = if sign == 1 { -abs_val } else { abs_val };

        if abs_val == 1 {
            num_eq1 += 1;
        } else {
            num_gt1 += 1;
        }
    }

    dec.trace.push(format!("{label} → coeffs = {:?}", coeffs));
    coeffs
}

/// I_4x4 MB path. Decodes:
///  - 16× prev_intra4x4_pred_mode_flag (FL, ctxIdxOffset=68, ctxIdxInc=0).
///  - rem_intra4x4_pred_mode (3-bin FL, ctxIdxOffset=69, ctxIdxInc=0) for each flag=0.
///  - intra_chroma_pred_mode (Table 9-34, ctxIdxOffset=64).
///  - coded_block_pattern (luma 4 bins + chroma 2 bins, § 9.3.3.1.1.4).
///  - mb_qp_delta (if any cbp bit set).
///  - Stops before residuals — reports what would follow.
fn decode_i4x4_mb(
    dec: &mut CabacDecoder,
    contexts: &mut [CabacContext; 1024],
    is_last: bool,
    prev_mb_qp_delta_nonzero: bool,
    cond_a: u32,
    cond_b: u32,
    neighbors: &mut CabacNeighborContext,
    mb_x: usize,
) -> (u8, CabacNeighborMB) {
    dec.trace.push("→ I_NxN (I_4x4)".into());

    // 16× prev_intra4x4_pred_mode_flag. ctxIdxOffset=68, ctxIdxInc=0.
    let mut prev_flags = [0u8; 16];
    let mut rem_modes = [-1i8; 16];
    for k in 0..16 {
        let flag = dec.decode_decision(
            &mut contexts[68],
            68,
            &format!("prev_intra4x4_pred_mode_flag[{k}]"),
        );
        prev_flags[k] = flag;
        if flag == 0 {
            // rem_intra4x4_pred_mode: 3 bins, each ctxIdxOffset=69, ctxIdxInc=0.
            // LSB-first — determined empirically by conformance testing
            // against a reference decoder. (Spec § 9.3.2.6 wording is
            // ambiguous; bit ordering fixed in Task #115 bug 3.)
            let b0 =
                dec.decode_decision(&mut contexts[69], 69, &format!("rem_intra4x4[{k}] bit0"));
            let b1 =
                dec.decode_decision(&mut contexts[69], 69, &format!("rem_intra4x4[{k}] bit1"));
            let b2 =
                dec.decode_decision(&mut contexts[69], 69, &format!("rem_intra4x4[{k}] bit2"));
            let rem = (b0 as i8) | ((b1 as i8) << 1) | ((b2 as i8) << 2);
            rem_modes[k] = rem;
        }
    }
    dec.trace.push(format!(
        "→ prev_flags = {:?}",
        prev_flags.iter().map(|f| *f as u32).collect::<Vec<_>>()
    ));
    dec.trace
        .push(format!("→ rem_modes (MSB-first recon) = {:?}", rem_modes));

    // intra_chroma_pred_mode (Table 9-34, § 9.3.3.1.1.8 eq 9-18).
    // ctxIdxOffset=64. bin 0 ctxIdxInc derives from neighbors' own
    // intra_chroma_pred_mode (NOT from mb_type cond_a/cond_b).
    let cpm_bin0_inc = ctx_idx_inc_intra_chroma_pred_mode_bin0(neighbors, mb_x);
    let _ = (cond_a, cond_b); // only used for mb_type earlier
    let cpm_bin0_ctx = 64 + cpm_bin0_inc;
    let cpm_bin0 = dec.decode_decision(
        &mut contexts[cpm_bin0_ctx as usize],
        cpm_bin0_ctx,
        &format!("chroma_pred bin0 (inc={cpm_bin0_inc})"),
    );
    let chroma_mode = if cpm_bin0 == 0 {
        0
    } else {
        let b1 = dec.decode_decision(&mut contexts[67], 67, "chroma_pred bin1");
        if b1 == 0 {
            1
        } else {
            let b2 = dec.decode_decision(&mut contexts[67], 67, "chroma_pred bin2");
            if b2 == 0 {
                2
            } else {
                3
            }
        }
    };
    dec.trace.push(format!("→ intra_chroma_pred_mode={chroma_mode}"));

    // coded_block_pattern: 4 luma 8×8 flags + 2 chroma bins.
    // § 9.3.3.1.1.4: ctxIdxOffset=73 for luma, ctxIdxOffset=77 for chroma.
    // Luma: per 8×8 block, ctxIdxInc derived from same-MB neighbors
    // (for interior blocks) and outer-MB neighbors (for edge blocks),
    // with progressively-built current-MB partial CBP folded in.
    let mut cbp_luma = [0u8; 4];
    let mut partial_cbp_luma = 0u8;
    for i in 0..4 {
        let inc = compute_cbp_luma_ctx_idx_inc_bin(i as u32, partial_cbp_luma, neighbors, mb_x);
        let ctx = 73 + inc;
        let flag = dec.decode_decision(
            &mut contexts[ctx as usize],
            ctx,
            &format!("cbp_luma[{i}] (inc={inc})"),
        );
        cbp_luma[i] = flag;
        if flag != 0 {
            partial_cbp_luma |= 1 << i;
        }
    }
    // Chroma CBP: 2 bins. bin0 signals "any chroma", bin1 signals AC.
    // ctxIdxOffset=77. ctxIdxInc per § 9.3.3.1.1.4 eq 9-11.
    let cpc_bin0_inc = ctx_idx_inc_cbp_chroma(neighbors, mb_x, 0);
    let cpc_bin0_ctx = 77 + cpc_bin0_inc;
    let cbp_chroma_bin0 = dec.decode_decision(
        &mut contexts[cpc_bin0_ctx as usize],
        cpc_bin0_ctx,
        &format!("cbp_chroma bin0 (inc={cpc_bin0_inc})"),
    );
    let cbp_chroma_bin1 = if cbp_chroma_bin0 == 1 {
        let cpc_bin1_inc = ctx_idx_inc_cbp_chroma(neighbors, mb_x, 1);
        let cpc_bin1_ctx = 77 + cpc_bin1_inc;
        Some(dec.decode_decision(
            &mut contexts[cpc_bin1_ctx as usize],
            cpc_bin1_ctx,
            &format!("cbp_chroma bin1 (inc={cpc_bin1_inc})"),
        ))
    } else {
        None
    };
    let cbp_chroma_code = cbp_chroma_bin0 + cbp_chroma_bin1.unwrap_or(0);
    dec.trace.push(format!(
        "→ cbp_luma={:?} cbp_chroma_code={cbp_chroma_code}",
        cbp_luma
    ));

    // mb_qp_delta — emitted only if CBP is nonzero.
    let any_cbp = cbp_luma.iter().any(|&f| f != 0) || cbp_chroma_code != 0;
    let mut qp_delta_nonzero_this_mb = false;
    if any_cbp {
        let qp_bin0_ctx: u32 = if prev_mb_qp_delta_nonzero { 61 } else { 60 };
        let mb_qp_delta_bin0 = dec.decode_decision(
            &mut contexts[qp_bin0_ctx as usize],
            qp_bin0_ctx,
            "mb_qp_delta bin0",
        );
        qp_delta_nonzero_this_mb = mb_qp_delta_bin0 != 0;
        if mb_qp_delta_bin0 == 1 {
            let mut n = 1u32;
            loop {
                let ctx_idx: u32 = if n == 1 { 62 } else { 63 };
                let b = dec.decode_decision(
                    &mut contexts[ctx_idx as usize],
                    ctx_idx,
                    "mb_qp_delta binN",
                );
                n += 1;
                if b == 0 {
                    break;
                }
                if n > 53 {
                    dec.trace.push("! mb_qp_delta overflow".into());
                    break;
                }
            }
            dec.trace.push(format!("→ mb_qp_delta unary length {n}"));
        }
    } else {
        dec.trace.push("→ cbp=0 so mb_qp_delta omitted".into());
    }

    // ── Residual emit (spec § 7.3.5.3) ──
    let block_index_to_pos: [(u8, u8); 16] = [
        (0, 0), (1, 0), (0, 1), (1, 1),
        (2, 0), (3, 0), (2, 1), (3, 1),
        (0, 2), (1, 2), (0, 3), (1, 3),
        (2, 2), (3, 2), (2, 3), (3, 3),
    ];
    let mut current_cbf = CurrentMbCbf::new();
    // I_4x4 is an intra MB type.
    let current_is_intra = true;
    for k in 0..16 {
        let (bx, by) = block_index_to_pos[k];
        if cbp_luma[k / 4] != 0 {
            let inc = compute_cbf_ctx_idx_inc_luma_4x4(&current_cbf, neighbors, mb_x, bx, by, current_is_intra);
            let coeffs = decode_residual_block(
                dec, contexts, 0, 15, 2, inc, &format!("luma4x4[k={k}, bx={bx}, by={by}]"),
            );
            current_cbf.set(2, block_pos_to_luma_idx(bx, by), coeffs.iter().any(|&v| v != 0));
        }
    }
    if cbp_chroma_code >= 1 {
        for plane in 0..2 {
            let inc = compute_cbf_ctx_idx_inc_chroma_dc(neighbors, mb_x, plane as u8, current_is_intra);
            let coeffs = decode_residual_block(
                dec, contexts, 0, 3, 3, inc, &format!("chroma_dc[plane={plane}]"),
            );
            current_cbf.set(3, plane, coeffs.iter().any(|&v| v != 0));
        }
    }
    if cbp_chroma_code == 2 {
        for plane in 0..2u8 {
            for sub in 0..4 {
                let bx = sub % 2;
                let by = sub / 2;
                let inc = compute_cbf_ctx_idx_inc_chroma_ac(
                    &current_cbf, neighbors, mb_x, plane, bx, by, current_is_intra,
                );
                let coeffs = decode_residual_block(
                    dec, contexts, 1, 14, 4, inc,
                    &format!("chroma_ac[plane={plane}, bx={bx}, by={by}]"),
                );
                current_cbf.set(
                    4,
                    phasm_core::codec::h264::cabac::neighbor::block_pos_to_chroma_ac_idx(plane, bx, by),
                    coeffs.iter().any(|&v| v != 0),
                );
            }
        }
    }

    let eos = dec.decode_terminate("end_of_slice");
    dec.trace.push(format!("→ end_of_slice={eos} (expected {})", is_last as u8));

    // Build the committed neighbor MB state.
    let mut nb = CabacNeighborMB::default();
    nb.mb_type = MbTypeClass::INxN;
    nb.intra_chroma_pred_mode = chroma_mode as u8;
    let cbp_luma_packed: u8 = (0..4)
        .map(|i| if cbp_luma[i] != 0 { 1u8 << i } else { 0 })
        .sum();
    nb.cbp_luma = cbp_luma_packed;
    nb.cbp_chroma = cbp_chroma_code as u8;
    nb.coded_block_flag_cat = current_cbf.to_neighbor_cbf();

    (if qp_delta_nonzero_this_mb { 1 } else { 0 }, nb)
}

fn decode_i16x16_mb(
    dec: &mut CabacDecoder,
    contexts: &mut [CabacContext; 1024],
    left_is_i_nxn_or_unavail: bool,
    top_is_i_nxn_or_unavail: bool,
    is_last: bool,
    prev_mb_qp_delta_nonzero: bool,
    neighbors: &mut CabacNeighborContext,
    mb_x: usize,
) -> (u8, CabacNeighborMB) {
    // mb_type (I-slice, § 9.3.3.1.1.3): binIdx 0 neighbor-derived.
    // condTermFlagN = 0 if N unavailable OR N is I_NxN; else 1.
    let cond_a: u32 = if left_is_i_nxn_or_unavail { 0 } else { 1 };
    let cond_b: u32 = if top_is_i_nxn_or_unavail { 0 } else { 1 };
    let bin0_inc = cond_a + cond_b;
    let bin0_ctx: u32 = 3 + bin0_inc;
    let bin0 = dec.decode_decision(&mut contexts[bin0_ctx as usize], bin0_ctx, "mb_type bin0");
    if bin0 == 0 {
        return decode_i4x4_mb(
            dec,
            contexts,
            is_last,
            prev_mb_qp_delta_nonzero,
            cond_a,
            cond_b,
            neighbors,
            mb_x,
        );
    }
    // binIdx 1: terminate.
    let bin1 = dec.decode_terminate("mb_type bin1 (term)");
    if bin1 == 1 {
        dec.trace.push("→ I_PCM".into());
        let mut nb = CabacNeighborMB::default();
        nb.mb_type = MbTypeClass::IPCM;
        return (0, nb);
    }
    // I_16x16 path. After state += 2 (spec convention), use ctxIdx 6, 7, 8, 9, 10.
    let bin2 = dec.decode_decision(&mut contexts[6], 6, "mb_type bin2 (cbp_luma)");
    let bin3 = dec.decode_decision(&mut contexts[7], 7, "mb_type bin3 (cbp_chroma test)");
    let bin4_opt = if bin3 == 1 {
        Some(dec.decode_decision(&mut contexts[8], 8, "mb_type bin4 (cbp_chroma bit)"))
    } else {
        None
    };
    let bin5 = dec.decode_decision(&mut contexts[9], 9, "mb_type bin5 (pred_mode MSB)");
    let bin6 = dec.decode_decision(&mut contexts[10], 10, "mb_type bin6 (pred_mode LSB)");

    let pred_mode = (bin5 << 1) | bin6;
    let cbp_luma = bin2;
    let cbp_chroma_code = bin3 + bin4_opt.unwrap_or(0);
    dec.trace.push(format!(
        "→ I_16x16 mode={pred_mode} cbp_luma={cbp_luma} cbp_chroma_code={cbp_chroma_code}"
    ));

    // intra_chroma_pred_mode: bin 0 neighbor-derived per § 9.3.3.1.1.8.
    let cpm_bin0_inc = ctx_idx_inc_intra_chroma_pred_mode_bin0(neighbors, mb_x);
    let cpm_bin0_ctx = 64 + cpm_bin0_inc;
    let cpm_bin0 = dec.decode_decision(
        &mut contexts[cpm_bin0_ctx as usize],
        cpm_bin0_ctx,
        &format!("chroma_pred bin0 (inc={cpm_bin0_inc})"),
    );
    let chroma_mode = if cpm_bin0 == 0 {
        0
    } else {
        let b1 = dec.decode_decision(&mut contexts[67], 67, "chroma_pred bin1");
        if b1 == 0 {
            1
        } else {
            let b2 = dec.decode_decision(&mut contexts[67], 67, "chroma_pred bin2");
            if b2 == 0 {
                2
            } else {
                3
            }
        }
    };
    dec.trace.push(format!("→ intra_chroma_pred_mode={chroma_mode}"));

    // mb_qp_delta — always emitted for I_16x16.
    // Bin 0 ctxIdxInc = 0 or 1 (based on prev_mb mb_qp_delta != 0).
    let qp_bin0_ctx: u32 = if prev_mb_qp_delta_nonzero { 61 } else { 60 };
    let mb_qp_delta_bin0 = dec.decode_decision(&mut contexts[qp_bin0_ctx as usize], qp_bin0_ctx, "mb_qp_delta bin0");
    let mut qp_delta_nonzero_this_mb = mb_qp_delta_bin0 != 0;
    if mb_qp_delta_bin0 == 1 {
        // More bins follow, unary binarization.
        let mut n = 1u32;
        loop {
            let ctx_idx: u32 = if n == 1 { 62 } else { 63 };
            let b = dec.decode_decision(&mut contexts[ctx_idx as usize], ctx_idx, "mb_qp_delta binN");
            n += 1;
            if b == 0 {
                break;
            }
            if n > 53 {
                dec.trace.push("! mb_qp_delta overflow".into());
                break;
            }
        }
        dec.trace.push(format!("→ mb_qp_delta unary length {n}"));
    } else {
        dec.trace.push("→ mb_qp_delta = 0".into());
        qp_delta_nonzero_this_mb = false;
    }

    // ── Residuals for I_16x16 ──
    let block_index_to_pos: [(u8, u8); 16] = [
        (0, 0), (1, 0), (0, 1), (1, 1),
        (2, 0), (3, 0), (2, 1), (3, 1),
        (0, 2), (1, 2), (0, 3), (1, 3),
        (2, 2), (3, 2), (2, 3), (3, 3),
    ];
    let mut current_cbf = CurrentMbCbf::new();
    // I_16x16 is an intra MB type.
    let current_is_intra = true;
    // Intra16x16DCLevel (cat=0, always emitted).
    let dc_inc = compute_cbf_ctx_idx_inc_luma_dc(neighbors, mb_x);
    let dc_coeffs = decode_residual_block(dec, contexts, 0, 15, 0, dc_inc, "i16x16_dc");
    current_cbf.set(0, 0, dc_coeffs.iter().any(|&v| v != 0));

    // 16 Intra16x16ACLevel blocks (cat=1), only if cbp_luma_flag=1.
    if cbp_luma == 1 {
        for k in 0..16 {
            let (bx, by) = block_index_to_pos[k];
            let inc = compute_cbf_ctx_idx_inc_luma_ac(&current_cbf, neighbors, mb_x, bx, by, current_is_intra);
            let coeffs = decode_residual_block(
                dec, contexts, 1, 15, 1, inc,
                &format!("i16x16_ac[k={k}, bx={bx}, by={by}]"),
            );
            current_cbf.set(1, block_pos_to_luma_idx(bx, by), coeffs.iter().any(|&v| v != 0));
        }
    }

    if cbp_chroma_code >= 1 {
        for plane in 0..2u8 {
            let inc = compute_cbf_ctx_idx_inc_chroma_dc(neighbors, mb_x, plane, current_is_intra);
            let coeffs = decode_residual_block(
                dec, contexts, 0, 3, 3, inc,
                &format!("i16x16_chroma_dc[plane={plane}]"),
            );
            current_cbf.set(3, plane as usize, coeffs.iter().any(|&v| v != 0));
        }
    }
    if cbp_chroma_code == 2 {
        for plane in 0..2u8 {
            for sub in 0..4 {
                let bx = sub % 2;
                let by = sub / 2;
                let inc = compute_cbf_ctx_idx_inc_chroma_ac(
                    &current_cbf, neighbors, mb_x, plane, bx, by, current_is_intra,
                );
                let coeffs = decode_residual_block(
                    dec, contexts, 1, 14, 4, inc,
                    &format!("i16x16_chroma_ac[plane={plane}, bx={bx}, by={by}]"),
                );
                current_cbf.set(
                    4,
                    phasm_core::codec::h264::cabac::neighbor::block_pos_to_chroma_ac_idx(plane, bx, by),
                    coeffs.iter().any(|&v| v != 0),
                );
            }
        }
    }

    let eos = dec.decode_terminate("end_of_slice");
    dec.trace.push(format!("→ end_of_slice={eos} (expected {})", is_last as u8));

    let mut nb = CabacNeighborMB::default();
    nb.mb_type = MbTypeClass::I16x16;
    nb.intra_chroma_pred_mode = chroma_mode as u8;
    nb.cbp_luma = if cbp_luma != 0 { 0x0F } else { 0 };
    nb.cbp_chroma = cbp_chroma_code as u8;
    nb.coded_block_flag_cat = current_cbf.to_neighbor_cbf();

    (if qp_delta_nonzero_this_mb { 1 } else { 0 }, nb)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <slice.h264>", args[0]);
        std::process::exit(2);
    }
    let bytes = fs::read(&args[1]).expect("read");
    let nals = parse_nal_units_annexb(&bytes).expect("parse nals");
    let sps_nal = nals.iter().find(|n| n.nal_type == NalType::SPS).expect("SPS");
    let pps_nal = nals.iter().find(|n| n.nal_type == NalType::PPS).expect("PPS");
    let sps = parse_sps(&sps_nal.rbsp).expect("parse sps");
    let pps = parse_pps(&pps_nal.rbsp).expect("parse pps");

    // Which slice to decode: set SLICE_INDEX env var (0-indexed, default 0).
    let slice_idx_env: usize = env::var("SLICE_INDEX").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let slice_nal = nals
        .iter()
        .filter(|n| matches!(n.nal_type, NalType::SLICE_IDR | NalType::SLICE))
        .nth(slice_idx_env)
        .expect("slice");
    let hdr = parse_slice_header(
        &slice_nal.rbsp,
        &sps,
        &pps,
        slice_nal.nal_type,
        slice_nal.nal_ref_idc,
    )
    .expect("slice header");
    eprintln!(
        "Slice type={:?} qp_delta={} data_bit_offset={}",
        hdr.slice_type, hdr.slice_qp_delta, hdr.data_bit_offset
    );

    let (rbsp, _ep_map) = remove_emulation_prevention_with_map(&slice_nal.rbsp);
    let slice_qp = (pps.pic_init_qp_minus26 as i32 + 26) + hdr.slice_qp_delta as i32;
    eprintln!("slice_qp = {slice_qp}");

    let slot = match hdr.slice_type {
        phasm_core::codec::h264::slice::SliceType::I
        | phasm_core::codec::h264::slice::SliceType::SI => CabacInitSlot::ISI,
        _ => CabacInitSlot::PIdc0,
    };
    let mut contexts = initialize_contexts(slot, slice_qp);

    // Spec § 9.3.1.2: cabac_alignment_one_bit before the first bin of
    // slice_data. It's already rbsp bits — skip past alignment.
    // The slice_data starts byte-aligned in CABAC. So bit position is
    // data_bit_offset rounded up to byte.
    let start_bit = (hdr.data_bit_offset as usize + 7) & !7;
    eprintln!("decoder start bit (byte-aligned) = {start_bit}");

    let mut dec = CabacDecoder::new(&rbsp, start_bit);
    eprintln!("after init: range=0x{:x} offset=0x{:x}", dec.range, dec.offset);

    // For a 32x32 I-only flat frame: 4 MBs.
    let mb_w = sps.pic_width_in_mbs as usize;
    let mb_h = sps.pic_height_in_map_units as usize;
    let total = mb_w * mb_h;
    eprintln!("Decoding {total} MBs");

    let is_p_slice = matches!(
        hdr.slice_type,
        phasm_core::codec::h264::slice::SliceType::P
            | phasm_core::codec::h264::slice::SliceType::SP,
    );

    let mut prev_qp_nonzero = false;
    let mut neighbors = CabacNeighborContext::new(mb_w, slot);
    for i in 0..total {
        let mb_x = i % mb_w;
        let mb_y = i / mb_w;
        if mb_x == 0 && mb_y > 0 {
            neighbors.new_row();
        }
        let is_last = i == total - 1;
        dec.trace.push(format!("\n=== MB ({mb_x},{mb_y}) ==="));

        if is_p_slice {
            // P-slice: mb_skip_flag first.
            let skip_inc = ctx_idx_inc_mb_skip_flag(&neighbors, mb_x);
            let skip_ctx: u32 = 11 + skip_inc;
            let skip_flag = dec.decode_decision(
                &mut contexts[skip_ctx as usize], skip_ctx, "mb_skip_flag",
            );
            if skip_flag == 1 {
                dec.trace.push("→ P_SKIP".into());
                let mut nb = CabacNeighborMB::default();
                nb.mb_type = MbTypeClass::PSkip;
                nb.mb_skip_flag = true;
                neighbors.commit(mb_x, nb);
                continue;
            }

            // mb_type prefix bin 0 at ctxIdxOffset=14: FIXED ctxIdxInc=0
            // per spec Table 9-39 (no neighbor derivation for this
            // offset). Task #21 root-cause fix 2026-04-23.
            let prefix_bin0_ctx: u32 = 14;
            let prefix_bin0 = dec.decode_decision(
                &mut contexts[prefix_bin0_ctx as usize],
                prefix_bin0_ctx,
                "mb_type_p prefix bin0",
            );
            if prefix_bin0 == 0 {
                // P-inter. Scope: this minimal oracle doesn't decode
                // P-inter residuals. Flag and stop.
                dec.trace.push(format!(
                    "→ P-inter at ({mb_x},{mb_y}) — oracle scope: stopping"
                ));
                break;
            }

            // Intra-in-P. Decode I-slice-like suffix at ctxIdxOffset=17.
            // Bin 0 (I_NxN vs I_16x16 indicator): per spec Table 9-41
            // the ctxIdxInc for offset=17 bin 0 is 0 unconditionally.
            let suffix_bin0_ctx: u32 = 17;
            let suffix_bin0 = dec.decode_decision(
                &mut contexts[suffix_bin0_ctx as usize], suffix_bin0_ctx,
                "mb_type_p suffix bin0 (I_NxN?)",
            );
            if suffix_bin0 == 0 {
                // I_NxN (I_4x4) in P-slice. Decode using the same
                // I-slice I_4x4 path; cond_a/cond_b are unused for
                // I_4x4 (only used for I_16x16 pred_mode context).
                let (qp_nz, committed) = decode_i4x4_mb(
                    &mut dec, &mut contexts, is_last, prev_qp_nonzero,
                    0, 0, &mut neighbors, mb_x,
                );
                prev_qp_nonzero = qp_nz != 0;
                neighbors.commit(mb_x, committed);
                continue;
            }
            // Bin 1: TERMINATE (I_PCM indicator).
            let bin1_term = dec.decode_terminate("mb_type_p suffix bin1 (term)");
            if bin1_term == 1 {
                dec.trace.push("→ I_PCM in P-slice".into());
                let mut nb = CabacNeighborMB::default();
                nb.mb_type = MbTypeClass::IPCM;
                neighbors.commit(mb_x, nb);
                continue;
            }
            // I_16x16 variant. Per Table 9-41 at ctxIdxOffset=17:
            //   bin 2: ctxIdxInc = 1  → ctx 18
            //   bin 3: ctxIdxInc = 2  → ctx 19
            //   bin 4: ctxIdxInc = 2 + b3 → ctx 19 or 20
            //   bin 5: ctxIdxInc = 3  → ctx 20
            //   bin 6: ctxIdxInc = 3  → ctx 20
            let bin2 = dec.decode_decision(
                &mut contexts[18], 18, "mb_type_p suffix bin2 (cbp_luma)",
            );
            let bin3 = dec.decode_decision(
                &mut contexts[19], 19, "mb_type_p suffix bin3 (cbp_chroma test)",
            );
            let bin4_opt = if bin3 == 1 {
                // MATCHING ENCODER's (17, 4) prior_bin rule:
                //   prior_bins[3] != 0 → ctxIdxInc = 2 (ctx 19)
                //   prior_bins[3] == 0 → ctxIdxInc = 3 (ctx 20)
                // We only emit bin4 when bin3 == 1, so bin3 != 0 → ctx 19.
                let ctx: u32 = 19;
                Some(dec.decode_decision(
                    &mut contexts[ctx as usize], ctx,
                    "mb_type_p suffix bin4 (cbp_chroma bit, enc-style)",
                ))
            } else { None };
            let bin5 = dec.decode_decision(
                &mut contexts[20], 20, "mb_type_p suffix bin5 (pred_mode MSB)",
            );
            let bin6 = dec.decode_decision(
                &mut contexts[20], 20, "mb_type_p suffix bin6 (pred_mode LSB)",
            );
            let pred_mode = (bin5 << 1) | bin6;
            let cbp_luma = bin2;
            let cbp_chroma_code = bin3 + bin4_opt.unwrap_or(0);
            dec.trace.push(format!(
                "→ I_16x16 (in P-slice) mode={pred_mode} cbp_luma={cbp_luma} cbp_chroma_code={cbp_chroma_code}"
            ));

            // intra_chroma_pred_mode + mb_qp_delta + residuals — reuse
            // the I-slice logic block. (This is the "body" part of
            // decode_i16x16_mb starting after bin 6.)
            let cpm_bin0_inc = ctx_idx_inc_intra_chroma_pred_mode_bin0(&neighbors, mb_x);
            let cpm_bin0_ctx = 64 + cpm_bin0_inc;
            let cpm_bin0 = dec.decode_decision(
                &mut contexts[cpm_bin0_ctx as usize], cpm_bin0_ctx,
                &format!("chroma_pred bin0 (inc={cpm_bin0_inc})"),
            );
            let chroma_mode = if cpm_bin0 == 0 { 0 } else {
                let b1 = dec.decode_decision(&mut contexts[67], 67, "chroma_pred bin1");
                if b1 == 0 { 1 } else {
                    let b2 = dec.decode_decision(&mut contexts[67], 67, "chroma_pred bin2");
                    if b2 == 0 { 2 } else { 3 }
                }
            };
            dec.trace.push(format!("→ intra_chroma_pred_mode={chroma_mode}"));

            // mb_qp_delta.
            let qp_bin0_ctx: u32 = if prev_qp_nonzero { 61 } else { 60 };
            let mb_qp_delta_bin0 = dec.decode_decision(
                &mut contexts[qp_bin0_ctx as usize], qp_bin0_ctx, "mb_qp_delta bin0",
            );
            let mut qp_delta_nonzero_this_mb = mb_qp_delta_bin0 != 0;
            if mb_qp_delta_bin0 == 1 {
                let mut n = 1u32;
                loop {
                    let ctx_idx: u32 = if n == 1 { 62 } else { 63 };
                    let b = dec.decode_decision(
                        &mut contexts[ctx_idx as usize], ctx_idx, "mb_qp_delta binN",
                    );
                    n += 1;
                    if b == 0 { break; }
                    if n > 53 { break; }
                }
                dec.trace.push(format!("→ mb_qp_delta unary length {n}"));
            } else {
                qp_delta_nonzero_this_mb = false;
                dec.trace.push("→ mb_qp_delta = 0".into());
            }

            // Residuals for I_16x16 in P-slice (same cat indices as I-slice).
            let block_index_to_pos: [(u8, u8); 16] = [
                (0, 0), (1, 0), (0, 1), (1, 1),
                (2, 0), (3, 0), (2, 1), (3, 1),
                (0, 2), (1, 2), (0, 3), (1, 3),
                (2, 2), (3, 2), (2, 3), (3, 3),
            ];
            let mut current_cbf = CurrentMbCbf::new();
            let current_is_intra = true;
            let dc_inc = compute_cbf_ctx_idx_inc_luma_dc(&neighbors, mb_x);
            let dc_coeffs = decode_residual_block(
                &mut dec, &mut contexts, 0, 15, 0, dc_inc, "i16x16_dc",
            );
            current_cbf.set(0, 0, dc_coeffs.iter().any(|&v| v != 0));

            if cbp_luma == 1 {
                for k in 0..16 {
                    let (bx, by) = block_index_to_pos[k];
                    let inc = compute_cbf_ctx_idx_inc_luma_ac(
                        &current_cbf, &neighbors, mb_x, bx, by, current_is_intra,
                    );
                    let coeffs = decode_residual_block(
                        &mut dec, &mut contexts, 1, 15, 1, inc,
                        &format!("i16x16_ac[k={k}, bx={bx}, by={by}]"),
                    );
                    current_cbf.set(1, block_pos_to_luma_idx(bx, by), coeffs.iter().any(|&v| v != 0));
                }
            }
            if cbp_chroma_code >= 1 {
                for plane in 0..2u8 {
                    let inc = compute_cbf_ctx_idx_inc_chroma_dc(
                        &neighbors, mb_x, plane, current_is_intra,
                    );
                    let coeffs = decode_residual_block(
                        &mut dec, &mut contexts, 0, 3, 3, inc,
                        &format!("i16x16_chroma_dc[plane={plane}]"),
                    );
                    current_cbf.set(3, plane as usize, coeffs.iter().any(|&v| v != 0));
                }
            }
            if cbp_chroma_code == 2 {
                for plane in 0..2u8 {
                    for sub in 0..4 {
                        let bx = sub % 2;
                        let by = sub / 2;
                        let inc = compute_cbf_ctx_idx_inc_chroma_ac(
                            &current_cbf, &neighbors, mb_x, plane, bx, by, current_is_intra,
                        );
                        let coeffs = decode_residual_block(
                            &mut dec, &mut contexts, 1, 14, 4, inc,
                            &format!("i16x16_chroma_ac[plane={plane}, bx={bx}, by={by}]"),
                        );
                        current_cbf.set(
                            4,
                            phasm_core::codec::h264::cabac::neighbor::block_pos_to_chroma_ac_idx(plane, bx, by),
                            coeffs.iter().any(|&v| v != 0),
                        );
                    }
                }
            }

            let eos = dec.decode_terminate("end_of_slice");
            dec.trace.push(format!("→ end_of_slice={eos} (expected {})", is_last as u8));

            let mut nb = CabacNeighborMB::default();
            nb.mb_type = MbTypeClass::I16x16;
            nb.intra_chroma_pred_mode = chroma_mode as u8;
            nb.cbp_luma = if cbp_luma != 0 { 0x0F } else { 0 };
            nb.cbp_chroma = cbp_chroma_code as u8;
            nb.coded_block_flag_cat = current_cbf.to_neighbor_cbf();
            neighbors.commit(mb_x, nb);
            prev_qp_nonzero = qp_delta_nonzero_this_mb;
        } else {
            // I-slice: existing path.
            let left_is_i_nxn_or_unavail = match neighbors.neighbor_a(mb_x) {
                None => true,
                Some(mb) => mb.mb_type == MbTypeClass::INxN,
            };
            let top_is_i_nxn_or_unavail = match neighbors.neighbor_b(mb_x) {
                None => true,
                Some(mb) => mb.mb_type == MbTypeClass::INxN,
            };
            let (qp_nz, committed) = decode_i16x16_mb(
                &mut dec,
                &mut contexts,
                left_is_i_nxn_or_unavail,
                top_is_i_nxn_or_unavail,
                is_last,
                prev_qp_nonzero,
                &mut neighbors,
                mb_x,
            );
            neighbors.commit(mb_x, committed);
            prev_qp_nonzero = qp_nz != 0;
        }
    }

    for line in &dec.trace {
        println!("{line}");
    }
}
