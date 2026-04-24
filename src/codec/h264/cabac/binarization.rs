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
}
