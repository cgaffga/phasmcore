// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 CABAC binarization primitives. Phase 6C.3.
//!
//! Converts syntax-element values to bin strings per spec § 9.3.2.
//! Six primitives cover every syntax element:
//!
//!  - Unary `U(v)` (§ 9.3.2.1)
//!  - Truncated Unary `TU(v, cMax)` (§ 9.3.2.2)
//!  - k-th order Exp-Golomb `EGk` (§ 9.3.2.3) — suffix only
//!  - Concatenated UEGk (§ 9.3.2.3)
//!  - Fixed-Length `FL(v, cMax)` (§ 9.3.2.6) — MSB-first
//!  - Table-driven for `mb_type`, `sub_mb_type`, `coded_block_pattern`
//!
//! The core primitives take a `&mut impl FnMut(u8)` sink so callers
//! can route bins directly to `encode_decision` (with per-bin ctxIdx)
//! or `encode_bypass` without intermediate allocation. Test helpers
//! wrap these to produce `Vec<u8>` for spec-vector assertions.
//!
//! Algorithm note:
//!   `docs/design/h264-encoder-algorithms/binarization.md`.

/// Unary binarization (spec § 9.3.2.1). Emits `v` ones followed by
/// a single zero.
pub fn binarize_unary(v: u32, emit: &mut impl FnMut(u8)) {
    for _ in 0..v {
        emit(1);
    }
    emit(0);
}

/// Truncated unary binarization (spec § 9.3.2.2). If `v < c_max`,
/// emits `U(v)`. If `v == c_max`, emits `c_max` ones with NO
/// trailing zero.
pub fn binarize_tu(v: u32, c_max: u32, emit: &mut impl FnMut(u8)) {
    debug_assert!(v <= c_max, "TU value {v} exceeds cMax {c_max}");
    if v < c_max {
        binarize_unary(v, emit);
    } else {
        for _ in 0..c_max {
            emit(1);
        }
    }
}

/// Fixed-length binarization (spec § 9.3.2.6). `fixedLength =
/// ceil(log2(cMax+1))`. **MSB-first** — `binIdx = 0` corresponds to
/// the MOST significant bit, with increasing binIdx toward the LSB.
/// Spec wording: "indexing of bins for the FL binarization is such
/// that binIdx = 0 relates to the most significant bit with
/// increasing values of binIdx towards the least significant bit".
pub fn binarize_fl(v: u32, c_max: u32, emit: &mut impl FnMut(u8)) {
    let fixed_length = fl_bit_length(c_max);
    if fixed_length == 0 {
        return;
    }
    for i in (0..fixed_length).rev() {
        emit(((v >> i) & 1) as u8);
    }
}

/// Compute the `fixedLength` for FL(cMax) per spec § 9.3.2.6:
/// `ceil(log2(cMax + 1))`.
#[inline]
pub fn fl_bit_length(c_max: u32) -> u32 {
    if c_max == 0 {
        0
    } else {
        32 - (c_max).leading_zeros()
    }
}

/// Emit the EGk suffix portion (spec § 9.3.2.3 bottom loop). Given
/// the residual `suf_s = |v| − uCoff` and starting `k`, produce the
/// Exp-Golomb suffix bins. Suffix bins are ALWAYS bypass-coded;
/// callers route them accordingly.
pub fn binarize_egk_suffix(mut suf_s: u32, mut k: u32, emit: &mut impl FnMut(u8)) {
    loop {
        if suf_s >= (1u32 << k) {
            emit(1);
            suf_s -= 1u32 << k;
            k += 1;
        } else {
            emit(0);
            // Unary terminator, then write k more binary digits.
            while k > 0 {
                k -= 1;
                emit(((suf_s >> k) & 1) as u8);
            }
            break;
        }
    }
}

/// Concatenated UEGk binarization (spec § 9.3.2.3). Two sinks so
/// the caller routes prefix bins (context-coded) separately from
/// suffix + sign bins (bypass-coded).
///
/// Parameters:
///  - `syn_el` — signed syntax-element value.
///  - `k` — EGk suffix order.
///  - `signed_val_flag` — true iff a sign bit follows when `v != 0`.
///  - `u_coff` — TU prefix cMax (equal to the suffix-boundary).
pub fn binarize_uegk(
    syn_el: i32,
    k: u32,
    signed_val_flag: bool,
    u_coff: u32,
    emit_prefix: &mut impl FnMut(u8),
    emit_suffix: &mut impl FnMut(u8),
) {
    let abs_v = syn_el.unsigned_abs();

    // Prefix: TU(min(u_coff, abs_v), u_coff). When abs_v == u_coff,
    // the TU is exactly u_coff ones (truncated, no trailing zero).
    let prefix_val = abs_v.min(u_coff);
    binarize_tu(prefix_val, u_coff, emit_prefix);

    // Suffix: only when |v| >= u_coff.
    if abs_v >= u_coff {
        let suf_s = abs_v - u_coff;
        binarize_egk_suffix(suf_s, k, emit_suffix);
    }

    // Sign: only when signed AND v != 0.
    if signed_val_flag && syn_el != 0 {
        emit_suffix(if syn_el > 0 { 0 } else { 1 });
    }
}

/// Remap `mb_qp_delta` per spec § 9.3.2.7 (Table 9-3): signed
/// value → unsigned mapped value. Then the binarization is plain
/// unary on the mapped value.
///
/// Mapping:
/// ```text
///  0 →  0
///  1 →  1     −1 →  2
///  2 →  3     −2 →  4
/// +n → 2n−1  −n → 2n
/// ```
#[inline]
pub fn mb_qp_delta_remap(qp_delta: i32) -> u32 {
    if qp_delta > 0 {
        (2 * qp_delta as u32) - 1
    } else {
        2 * qp_delta.unsigned_abs()
    }
}

// ─── Table-driven: mb_type (Table 9-36 I-slice) ─────────────────

/// Bin string for an I-slice mb_type value (0..=25). Spec Table 9-36.
/// Returns a slice of bins. Bin 1 of I_PCM (mb_type=25) is
/// terminate-coded — caller must route that bin via
/// `encode_terminate`, not `encode_decision`.
pub fn mb_type_i_bins(mb_type: u32) -> &'static [u8] {
    debug_assert!(mb_type <= 25, "I-slice mb_type {mb_type} out of range");
    MB_TYPE_I_BINS[mb_type as usize]
}

/// Full Table 9-36 — I-slice mb_type → bin string.
const MB_TYPE_I_BINS: &[&[u8]] = &[
    // 0: I_NxN
    &[0],
    // 1..4: I_16x16_<mode>_0_0 — 6 bins
    &[1, 0, 0, 0, 0, 0],
    &[1, 0, 0, 0, 0, 1],
    &[1, 0, 0, 0, 1, 0],
    &[1, 0, 0, 0, 1, 1],
    // 5..12: I_16x16_<mode>_1_0 / _2_0 — 7 bins
    &[1, 0, 0, 1, 0, 0, 0],
    &[1, 0, 0, 1, 0, 0, 1],
    &[1, 0, 0, 1, 0, 1, 0],
    &[1, 0, 0, 1, 0, 1, 1],
    &[1, 0, 0, 1, 1, 0, 0],
    &[1, 0, 0, 1, 1, 0, 1],
    &[1, 0, 0, 1, 1, 1, 0],
    &[1, 0, 0, 1, 1, 1, 1],
    // 13..16: I_16x16_<mode>_0_1 — 6 bins
    &[1, 0, 1, 0, 0, 0],
    &[1, 0, 1, 0, 0, 1],
    &[1, 0, 1, 0, 1, 0],
    &[1, 0, 1, 0, 1, 1],
    // 17..24: I_16x16_<mode>_1_1 / _2_1 — 7 bins
    &[1, 0, 1, 1, 0, 0, 0],
    &[1, 0, 1, 1, 0, 0, 1],
    &[1, 0, 1, 1, 0, 1, 0],
    &[1, 0, 1, 1, 0, 1, 1],
    &[1, 0, 1, 1, 1, 0, 0],
    &[1, 0, 1, 1, 1, 0, 1],
    &[1, 0, 1, 1, 1, 1, 0],
    &[1, 0, 1, 1, 1, 1, 1],
    // 25: I_PCM — 2 bins; bin 1 is terminate-coded.
    &[1, 1],
];

// ─── Table-driven: mb_type (Table 9-37 P-slice P-rows) ──────────

/// Bin string for a P-slice mb_type value. Spec Table 9-37 P rows.
///
/// For values 0..3: returns the P-partition bin string (3 bins).
/// For value 4 (P_8x8ref0): FORBIDDEN in CABAC — this function
/// panics in debug, returns `&[]` in release.
/// For values 5..30: returns the 1-bit prefix; caller appends
/// `mb_type_i_bins(value - 5)` as suffix.
pub fn mb_type_p_bins_prefix(mb_type: u32) -> &'static [u8] {
    match mb_type {
        0 => &[0, 0, 0], // P_L0_16x16
        1 => &[0, 1, 1], // P_L0_L0_16x8
        2 => &[0, 1, 0], // P_L0_L0_8x16
        3 => &[0, 0, 1], // P_8x8
        4 => {
            debug_assert!(false, "P_8x8ref0 (mb_type=4) is forbidden in CABAC");
            &[]
        }
        _ => &[1], // Intra-in-P prefix; suffix = mb_type_i_bins(mb_type - 5)
    }
}

// ─── Table-driven: mb_type (Table 9-37 B-slice rows) ────────────
//
// Phase 6E-A3 — B-slice mb_type bin tree per H.264 Spec Table 9-37
// (B rows). The encoder-side bin tree below is the inverse of the
// spec's parsing tree.
//
// Numeric mb_type values per spec Table 7-14:
//   0  = B_Direct_16x16
//   1  = B_L0_16x16
//   2  = B_L1_16x16
//   3  = B_Bi_16x16
//   4  = B_L0_L0_16x8     5  = B_L0_L0_8x16
//   6  = B_L1_L1_16x8     7  = B_L1_L1_8x16
//   8  = B_L0_L1_16x8     9  = B_L0_L1_8x16
//   10 = B_L1_L0_16x8     11 = B_L1_L0_8x16
//   12 = B_L0_Bi_16x8     13 = B_L0_Bi_8x16
//   14 = B_L1_Bi_16x8     15 = B_L1_Bi_8x16
//   16 = B_Bi_L0_16x8     17 = B_Bi_L0_8x16
//   18 = B_Bi_L1_16x8     19 = B_Bi_L1_8x16
//   20 = B_Bi_Bi_16x8     21 = B_Bi_Bi_8x16
//   22 = B_8x8
//   23+ = I-slice mb_types (intra-in-B), suffix `mb_type_i_bins(value - 23)`.
//
// Phase 6E-A3 ships values 0..3 + 22 (16x16 partitions + B_8x8) +
// 23+ for intra-in-B fallback. Values 4..21 (16x8 / 8x16 partitions)
// are deferred to §6E-A6.

/// Bin string for a B-slice `mb_type` value. Spec Table 9-37 B rows.
///
/// For values 0..21: returns the full B-partition bin string (1..6 bins).
/// For value 22: returns the 6-bin tag for `B_8x8`.
/// For values 23..47: returns the 6-bin intra-in-B prefix; caller
/// appends `mb_type_i_bins(value - 23)` as suffix.
///
/// Bin-by-bin tree (per spec Table 9-37 B rows):
/// ```text
///   bin0 = 0            → B_Direct_16x16 (value 0)
///   bin0 = 1, bin1 = 0  → 16x16 L0/L1; bin2 picks
///                            bin2=0 → B_L0_16x16  (value 1)
///                            bin2=1 → B_L1_16x16  (value 2)
///   bin0 = 1, bin1 = 1  → multi-partition; 4 more bins build a value v:
///                            v = (bin2<<3)|(bin3<<2)|(bin4<<1)|bin5
///                            v ∈ [0,7]   → mb_type = v + 3   (3..10)
///                            v == 13     → intra-in-B: bin6.. = mb_type_i_bins(value - 23)
///                            v == 14     → mb_type = 11
///                            v == 15     → mb_type = 22 (B_8x8)
///                            else        → mb_type = (v<<1 | bin6) - 4
///                                          (covers values 12..21)
/// ```
///
/// Phase 6E-A3 implementation note: rather than walk the tree above
/// at every emit site, we precompute the full bin string per value
/// in `MB_TYPE_B_BINS` and return a slice into that table.
pub fn mb_type_b_bins(mb_type: u32) -> &'static [u8] {
    debug_assert!(mb_type <= 22, "B-slice non-intra mb_type must be 0..=22");
    MB_TYPE_B_BINS[mb_type as usize]
}

/// Intra-in-B prefix: B-slice mb_type ≥ 23 means intra. The encoder
/// emits this 6-bin prefix `[1, 1, 1, 1, 0, 1]` (= bin pattern for
/// value 13 in the multi-partition tree) and then the I-slice
/// suffix via `mb_type_i_bins(mb_type - 23)`.
///
/// Wire-level: a conforming decoder recognizes value 13 as the
/// "intra branch" and dispatches into the I-slice mb_type decode path.
pub fn mb_type_b_intra_prefix() -> &'static [u8] {
    &[1, 1, 1, 1, 0, 1]
}

/// Full Table 9-37 B-slice mb_type → bin string for values 0..=22.
/// Each row encodes the bin sequence emitted/decoded by the
/// spec-defined tree above.
const MB_TYPE_B_BINS: [&[u8]; 23] = [
    &[0],                            //  0: B_Direct_16x16 (bin0=0 short-circuit)
    &[1, 0, 0],                      //  1: B_L0_16x16     (bin0=1, bin1=0, bin2=0)
    &[1, 0, 1],                      //  2: B_L1_16x16     (bin0=1, bin1=0, bin2=1)
    &[1, 1, 0, 0, 0, 0],             //  3: B_Bi_16x16     (mb_type-3 == 0 → v=0)
    &[1, 1, 0, 0, 0, 1],             //  4: B_L0_L0_16x8   (v=1)
    &[1, 1, 0, 0, 1, 0],             //  5: B_L0_L0_8x16   (v=2)
    &[1, 1, 0, 0, 1, 1],             //  6: B_L1_L1_16x8   (v=3)
    &[1, 1, 0, 1, 0, 0],             //  7: B_L1_L1_8x16   (v=4)
    &[1, 1, 0, 1, 0, 1],             //  8: B_L0_L1_16x8   (v=5)
    &[1, 1, 0, 1, 1, 0],             //  9: B_L0_L1_8x16   (v=6)
    &[1, 1, 0, 1, 1, 1],             // 10: B_L1_L0_16x8   (v=7)
    // Values 11..21 use the (v<<1 | bin6) - 4 path; v ∈ {8..12}.
    // mb_type = (v<<1 | bin6) - 4. So mb_type=11 ↔ v=8, bin6=1 ↔ raw=17 → 17-4=13? No.
    // Per spec Table 9-37: for v in {8,9,10,11,12} (NOT 13/14/15), 5th bin is read
    //   mb_type = ((v<<1) | bin6) - 4
    // v=8 → 16|bin6 → bin6=0 ⇒ 16-4=12 ; bin6=1 ⇒ 17-4=13
    // We dispatch v=13→intra (above), so bin6 path covers raw mb_type ∈ {12..21}.
    // value→(v,bin6):
    //   12→(8,0)  13→(8,1) → INTRA via v=13 path? No, the intra branch is v==13
    //   in the 4-bit value. Here we map mb_type to raw_v_bin6 = mb_type+4:
    //   mb_type=12 → raw=16 → v=8, bin6=0 → bins [1,1,0,1, 0,0,0]
    //   mb_type=13 → raw=17 → v=8, bin6=1 → bins [1,1,0,1, 0,0,1]
    //   mb_type=14 → raw=18 → v=9, bin6=0 → bins [1,1,0,1, 0,1,0]
    //   mb_type=15 → raw=19 → v=9, bin6=1 → bins [1,1,0,1, 0,1,1]
    //   mb_type=16 → raw=20 → v=10, bin6=0 → bins [1,1,0,1, 1,0,0]
    //   mb_type=17 → raw=21 → v=10, bin6=1 → bins [1,1,0,1, 1,0,1]
    //   mb_type=18 → raw=22 → v=11, bin6=0 → bins [1,1,0,1, 1,1,0]
    //   mb_type=19 → raw=23 → v=11, bin6=1 → bins [1,1,0,1, 1,1,1]
    //   mb_type=20 → raw=24 → v=12, bin6=0 → bins [1,1,1,0, 0,0,0]
    //   mb_type=21 → raw=25 → v=12, bin6=1 → bins [1,1,1,0, 0,0,1]
    // (where the 4-bit prefix is v=8..12 = 1000..1100, hence the 5-bit prefix
    //  contributes [1,0,0,0]..[1,1,0,0] AFTER the leading [1,1] header.)
    //
    // For Phase 6E-A3 we ship 0..3 + 22 (and intra via prefix). Values
    // 4..21 are populated for completeness but the encoder won't emit
    // them until §6E-A6. Decoder won't see them either if we only emit
    // §6E-A3-supported values.
    &[1, 1, 0, 1, 0, 0, 0],          // 11: B_L0_L1_8x16 actually — placeholder for §6E-A6
    &[1, 1, 0, 1, 0, 0, 0],          // 12: placeholder
    &[1, 1, 0, 1, 0, 0, 1],          // 13: placeholder
    &[1, 1, 0, 1, 0, 1, 0],          // 14: placeholder
    &[1, 1, 0, 1, 0, 1, 1],          // 15: placeholder
    &[1, 1, 0, 1, 1, 0, 0],          // 16: placeholder
    &[1, 1, 0, 1, 1, 0, 1],          // 17: placeholder
    &[1, 1, 0, 1, 1, 1, 0],          // 18: placeholder
    &[1, 1, 0, 1, 1, 1, 1],          // 19: placeholder
    &[1, 1, 1, 0, 0, 0, 0],          // 20: placeholder
    &[1, 1, 1, 0, 0, 0, 1],          // 21: placeholder
    &[1, 1, 1, 1, 1, 1],             // 22: B_8x8 (v=15 short-circuit, no bin6)
];

// ─── Table-driven: sub_mb_type (Table 9-38 P rows) ──────────────

/// Bin string for a P-slice sub_mb_type value (0..=3). Spec Table 9-38.
pub fn sub_mb_type_p_bins(sub_mb_type: u32) -> &'static [u8] {
    match sub_mb_type {
        0 => &[1],       // P_L0_8x8
        1 => &[0, 0],    // P_L0_8x4
        2 => &[0, 1, 1], // P_L0_4x8
        3 => &[0, 1, 0], // P_L0_4x4
        _ => {
            debug_assert!(false, "P-slice sub_mb_type {sub_mb_type} out of range");
            &[]
        }
    }
}

// ─── Test helpers (Vec<u8> wrappers) ────────────────────────────

/// Unary binarization → `Vec<u8>` (test / debug convenience).
pub fn unary_to_bins(v: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((v + 1) as usize);
    binarize_unary(v, &mut |b| out.push(b));
    out
}

/// Truncated unary → `Vec<u8>`.
pub fn tu_to_bins(v: u32, c_max: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((c_max + 1) as usize);
    binarize_tu(v, c_max, &mut |b| out.push(b));
    out
}

/// Fixed-length → `Vec<u8>` (LSB-first).
pub fn fl_to_bins(v: u32, c_max: u32) -> Vec<u8> {
    let mut out = Vec::new();
    binarize_fl(v, c_max, &mut |b| out.push(b));
    out
}

/// UEGk → two `Vec<u8>` (prefix, suffix_and_sign).
pub fn uegk_to_bins(
    syn_el: i32,
    k: u32,
    signed_val_flag: bool,
    u_coff: u32,
) -> (Vec<u8>, Vec<u8>) {
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();
    binarize_uegk(
        syn_el,
        k,
        signed_val_flag,
        u_coff,
        &mut |b| prefix.push(b),
        &mut |b| suffix.push(b),
    );
    (prefix, suffix)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Unary ──────────────────────────────────────────────────

    #[test]
    fn unary_spec_examples() {
        assert_eq!(unary_to_bins(0), vec![0]);
        assert_eq!(unary_to_bins(1), vec![1, 0]);
        assert_eq!(unary_to_bins(2), vec![1, 1, 0]);
        assert_eq!(unary_to_bins(3), vec![1, 1, 1, 0]);
        assert_eq!(unary_to_bins(5), vec![1, 1, 1, 1, 1, 0]);
    }

    #[test]
    fn unary_length_is_v_plus_one() {
        for v in 0..20 {
            assert_eq!(unary_to_bins(v).len(), (v + 1) as usize);
        }
    }

    // ─── Truncated Unary ────────────────────────────────────────

    #[test]
    fn tu_below_cmax_matches_unary() {
        assert_eq!(tu_to_bins(0, 3), vec![0]);
        assert_eq!(tu_to_bins(1, 3), vec![1, 0]);
        assert_eq!(tu_to_bins(2, 3), vec![1, 1, 0]);
    }

    #[test]
    fn tu_at_cmax_truncates_no_trailing_zero() {
        // v == cMax → cMax ones, no trailing 0.
        assert_eq!(tu_to_bins(3, 3), vec![1, 1, 1]);
        assert_eq!(tu_to_bins(5, 5), vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn tu_cmax_3_intra_chroma_pred_mode() {
        // Used for intra_chroma_pred_mode.
        // DC=0 → "0", H=1 → "10", V=2 → "110", Plane=3 → "111".
        assert_eq!(tu_to_bins(0, 3), vec![0]);
        assert_eq!(tu_to_bins(1, 3), vec![1, 0]);
        assert_eq!(tu_to_bins(2, 3), vec![1, 1, 0]);
        assert_eq!(tu_to_bins(3, 3), vec![1, 1, 1]);
    }

    // ─── Fixed-Length ───────────────────────────────────────────

    #[test]
    fn fl_bit_length_spec_examples() {
        assert_eq!(fl_bit_length(0), 0);
        assert_eq!(fl_bit_length(1), 1);
        assert_eq!(fl_bit_length(3), 2);
        assert_eq!(fl_bit_length(7), 3);
        assert_eq!(fl_bit_length(15), 4);
    }

    #[test]
    fn fl_is_msb_first() {
        // Spec § 9.3.2.6: binIdx 0 = MSB.
        // cMax=15 → 4 bits. v=5 = 0b0101 MSB-first → [0, 1, 0, 1].
        assert_eq!(fl_to_bins(5, 15), vec![0, 1, 0, 1]);
        // v=7 = 0b111 → [1, 1, 1] at cMax=7.
        assert_eq!(fl_to_bins(7, 7), vec![1, 1, 1]);
        // v=2 = 0b10 → [1, 0] at cMax=3 (MSB-first).
        assert_eq!(fl_to_bins(2, 3), vec![1, 0]);
        // v=4 = 0b100 → [1, 0, 0] at cMax=7 (MSB-first).
        assert_eq!(fl_to_bins(4, 7), vec![1, 0, 0]);
    }

    #[test]
    fn fl_1bit_flags() {
        assert_eq!(fl_to_bins(0, 1), vec![0]);
        assert_eq!(fl_to_bins(1, 1), vec![1]);
    }

    // ─── UEGk: UEG0 (coeff_abs_level_minus1) ───────────────────

    #[test]
    fn uegk_uego_small_values() {
        let (p, s) = uegk_to_bins(0, 0, false, 14);
        assert_eq!(p, vec![0]);
        assert!(s.is_empty());
        let (p, s) = uegk_to_bins(13, 0, false, 14);
        assert_eq!(p, vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]);
        assert!(s.is_empty());
    }

    #[test]
    fn uegk_uego_at_ucoff_boundary() {
        // v=14 → prefix 14 ones (truncated), suffix sufS=0 k=0 → "0".
        let (p, s) = uegk_to_bins(14, 0, false, 14);
        assert_eq!(p, vec![1; 14]);
        assert_eq!(s, vec![0]);
    }

    #[test]
    fn uegk_uego_above_ucoff() {
        // v=15 → prefix 14 ones, suffix sufS=1 k=0:
        //   iter 1: 1 >= 1, emit 1, sufS=0, k=1.
        //   iter 2: 0 < 2, emit 0, write 1 bit of 0 → "0".
        // Suffix = [1, 0, 0].
        let (p, s) = uegk_to_bins(15, 0, false, 14);
        assert_eq!(p, vec![1; 14]);
        assert_eq!(s, vec![1, 0, 0]);
    }

    // ─── UEGk: UEG3 (mvd) ──────────────────────────────────────

    #[test]
    fn uegk_uge3_zero() {
        let (p, s) = uegk_to_bins(0, 3, true, 9);
        assert_eq!(p, vec![0]); // no sign for v=0
        assert!(s.is_empty());
    }

    #[test]
    fn uegk_uge3_positive_small() {
        // v=1 → prefix "10", sign 0. All in their own sinks.
        let (p, s) = uegk_to_bins(1, 3, true, 9);
        assert_eq!(p, vec![1, 0]);
        assert_eq!(s, vec![0]); // sign
    }

    #[test]
    fn uegk_uge3_negative_small() {
        let (p, s) = uegk_to_bins(-3, 3, true, 9);
        assert_eq!(p, vec![1, 1, 1, 0]);
        assert_eq!(s, vec![1]); // sign: negative
    }

    #[test]
    fn uegk_uge3_at_ucoff() {
        // v=9 → prefix = 9 ones (truncated). Suffix: sufS=0, k=3.
        //   iter 1: 0 < 8, emit 0. Write 3 bits of 0 → "000".
        // Suffix = [0, 0, 0, 0]. Sign = [0].
        let (p, s) = uegk_to_bins(9, 3, true, 9);
        assert_eq!(p, vec![1; 9]);
        assert_eq!(s, vec![0, 0, 0, 0, 0]); // EGk + sign
    }

    #[test]
    fn uegk_uge3_above_ucoff() {
        // v=17 → prefix = 9 ones (truncated TU).
        // Suffix: sufS=8, k=3.
        //   iter 1: 8 >= 8 → emit 1, sufS=0, k=4.
        //   iter 2: 0 < 16 → emit 0 (terminator), write 4 bits of
        //           sufS=0 MSB-first → 0,0,0,0.
        // EG bins: [1, 0, 0, 0, 0, 0]  (6 bins)
        // Sign (v=17 > 0): 0.
        // Suffix total: 7 bins.
        let (p, s) = uegk_to_bins(17, 3, true, 9);
        assert_eq!(p, vec![1; 9]);
        assert_eq!(s, vec![1, 0, 0, 0, 0, 0, 0]);
    }

    // ─── mb_qp_delta remap ─────────────────────────────────────

    #[test]
    fn mb_qp_delta_remap_spec_examples() {
        assert_eq!(mb_qp_delta_remap(0), 0);
        assert_eq!(mb_qp_delta_remap(1), 1);
        assert_eq!(mb_qp_delta_remap(-1), 2);
        assert_eq!(mb_qp_delta_remap(2), 3);
        assert_eq!(mb_qp_delta_remap(-2), 4);
        assert_eq!(mb_qp_delta_remap(5), 9);
        assert_eq!(mb_qp_delta_remap(-5), 10);
    }

    // ─── mb_type tables ─────────────────────────────────────────

    #[test]
    fn mb_type_i_all_values_have_bins() {
        for v in 0..=25 {
            let bins = mb_type_i_bins(v);
            assert!(!bins.is_empty(), "mb_type_i_bins({v}) empty");
        }
    }

    #[test]
    fn mb_type_i_spec_fixed_points() {
        assert_eq!(mb_type_i_bins(0), &[0][..]);
        assert_eq!(mb_type_i_bins(1), &[1, 0, 0, 0, 0, 0][..]);
        assert_eq!(mb_type_i_bins(2), &[1, 0, 0, 0, 0, 1][..]);
        assert_eq!(mb_type_i_bins(24), &[1, 0, 1, 1, 1, 1, 1][..]);
        assert_eq!(mb_type_i_bins(25), &[1, 1][..]);
    }

    #[test]
    fn mb_type_p_prefix_spec_values() {
        assert_eq!(mb_type_p_bins_prefix(0), &[0, 0, 0][..]);
        assert_eq!(mb_type_p_bins_prefix(1), &[0, 1, 1][..]);
        assert_eq!(mb_type_p_bins_prefix(2), &[0, 1, 0][..]);
        assert_eq!(mb_type_p_bins_prefix(3), &[0, 0, 1][..]);
        // Intra-in-P: prefix = '1'.
        assert_eq!(mb_type_p_bins_prefix(5), &[1][..]);
        assert_eq!(mb_type_p_bins_prefix(30), &[1][..]);
    }

    #[test]
    fn sub_mb_type_p_spec_values() {
        assert_eq!(sub_mb_type_p_bins(0), &[1][..]);
        assert_eq!(sub_mb_type_p_bins(1), &[0, 0][..]);
        assert_eq!(sub_mb_type_p_bins(2), &[0, 1, 1][..]);
        assert_eq!(sub_mb_type_p_bins(3), &[0, 1, 0][..]);
    }

    /// §6E-A3 — B-slice mb_type bin tree spec fixed points. The
    /// values 0..3 + 22 are the §6E-A3 active set; 4..21 are
    /// placeholders pending §6E-A6.
    #[test]
    fn mb_type_b_bins_active_set() {
        // Direct → bin 0 short-circuit.
        assert_eq!(mb_type_b_bins(0), &[0][..]);
        // 16x16 L0/L1 → 3 bins each.
        assert_eq!(mb_type_b_bins(1), &[1, 0, 0][..]);
        assert_eq!(mb_type_b_bins(2), &[1, 0, 1][..]);
        // 16x16 Bi (mb_type=3) → 6-bin multi-partition with v=0.
        assert_eq!(mb_type_b_bins(3), &[1, 1, 0, 0, 0, 0][..]);
        // B_8x8 (mb_type=22) → 6-bin v=15 short-circuit.
        assert_eq!(mb_type_b_bins(22), &[1, 1, 1, 1, 1, 1][..]);
    }

    /// §6E-A3 — intra-in-B prefix is the v=13 pattern in the spec tree.
    #[test]
    fn mb_type_b_intra_prefix_matches_v13() {
        assert_eq!(mb_type_b_intra_prefix(), &[1, 1, 1, 1, 0, 1][..]);
    }

    /// §6E-A3 — every B mb_type value 0..=22 returns a non-empty
    /// bin string (sanity).
    #[test]
    fn mb_type_b_bins_all_non_empty() {
        for v in 0..=22u32 {
            let bins = mb_type_b_bins(v);
            assert!(!bins.is_empty(), "mb_type_b_bins({v}) empty");
        }
    }

    /// §6E-A3 — the 16x16 family (values 1..=3) starts with bin0=1
    /// (non-Direct) and a 0-bin in position 1 OR position 2-3
    /// per the spec Table 9-37 tree structure.
    #[test]
    fn mb_type_b_bins_16x16_family_starts_correctly() {
        // L0_16x16, L1_16x16: bin0=1, bin1=0
        assert_eq!(mb_type_b_bins(1)[0], 1);
        assert_eq!(mb_type_b_bins(1)[1], 0);
        assert_eq!(mb_type_b_bins(2)[0], 1);
        assert_eq!(mb_type_b_bins(2)[1], 0);
        // Bi_16x16: bin0=1, bin1=1 (multi-partition path)
        assert_eq!(mb_type_b_bins(3)[0], 1);
        assert_eq!(mb_type_b_bins(3)[1], 1);
    }
}
