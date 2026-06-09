// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Portions translated from `naoto256/jpeg-rusturbo` v0.3.0
// (`src/arch/neon.rs::fdct_islow_inner`, MIT OR Apache-2.0),
// itself a Rust port of libjpeg-turbo `simd/arm/jfdctint-neon.c`
// (BSD-3-Clause AND IJG). The phasm port follows libjpeg-turbo's
// upstream structure; jpeg-rusturbo serves as a reference for the
// Rust-intrinsic mechanics.
//
// "This software is based in part on the work of the Independent
//  JPEG Group" — IJG license clause 2.

//! Per-architecture SIMD kernels for the LL&M integer DCT/IDCT.
//!
//! ## NEON (aarch64)
//!
//! Both forward and inverse DCT NEON kernels are wired in. The IDCT
//! was written directly from libjpeg-turbo upstream
//! `simd/arm/jidctint-neon.c` since jpeg-rusturbo v0.3.0 doesn't have
//! a NEON IDCT (their `idct_islow` falls back to scalar on aarch64).
//!
//! - **fdct** uses 8-wide (q-register) vectors throughout — pure i16
//!   arithmetic with only short-lived i32x4 widening at multiplies.
//! - **idct** uses 4-wide (d-register) vectors and processes left/right
//!   4×8 halves in sequence (matching libjpeg-turbo upstream). 8-wide
//!   isn't viable here: each i16x8 "value" carries (i32x4_lo,i32x4_hi)
//!   through the IDCT, and ~15 intermediates × 2 = 30 q-registers,
//!   exceeding the 32-register file even on modern aarch64.
//!
//! ## IDCT i16 input envelope (binding precondition)
//!
//! The IDCT NEON kernel — matching libjpeg-turbo upstream — uses **i16
//! arithmetic throughout pass 2**, including `vadd_s16(col7, col3)`
//! and `vadd_s16(col5, col1)` for the odd-part pre-sums. These
//! non-saturating adds wrap silently if the operand sum exceeds
//! ±32767.
//!
//! **Precondition**: the i16 workspace values from pass 1 must satisfy
//! `|workspace[r,c1]| + |workspace[r,c2]| ≤ i16::MAX` for each
//! odd-part-summed pair (c1, c2) ∈ {(7,3), (5,1)}. For valid 8-bit
//! JPEG cover images this holds with orders of magnitude of margin —
//! verified by 45 image-stego integration tests (armor_roundtrip,
//! ghost_roundtrip, si_uniward, optimizer_*) all green under the NEON
//! IDCT.
//!
//! Synthetic q values larger than ~[-32, 32] AC can violate the bound
//! and produce NEON-vs-scalar divergence. The unit tests
//! `idct_neon_bitexact_vs_scalar` and `idct_parity_random_blocks` are
//! deliberately scoped to this envelope.
//!
//! ## Determinism (binding spec)
//!
//! - Pure integer arithmetic. No f64. No FMA. No SIMD rounding-mode
//!   dependence.
//! - Bit-identical output across scalar / NEON / AVX2 / WASM SIMD
//!   paths, verified by `tests/pixels_aan_cross_platform.rs`.
//! - Parity envelope vs the f64 reference: ≤ 1 LSB per coefficient on
//!   forward, ≤ 1 pixel-unit per pixel on inverse.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "aarch64")]
pub(super) mod neon {
    use core::arch::aarch64::*;

    /// Q13 constants packed for `vmull_lane_s16` / `vmlal_lane_s16`.
    /// Loaded once at the top of the kernel into three 4-lane registers
    /// (`consts1`, `consts2`, `consts3`).
    ///
    /// Lane map (constN.<L> = pattern):
    ///   consts1 = [ F_0_298,    -F_0_390,    F_0_541,    F_0_765 ]
    ///   consts2 = [-F_0_899,     F_1_175,    F_1_501,   -F_1_847 ]
    ///   consts3 = [-F_1_961,     F_2_053,   -F_2_562,    F_3_072 ]
    ///
    /// Matches the layout of libjpeg-turbo `jsimd_fdct_islow_neon_consts`.
    static FDCT_CONSTS: [i16; 12] = [
        2446, -3196, 4433, 6270, // F_0_298, -F_0_390,  F_0_541,  F_0_765
        -7373, 9633, 12299, -15137, // -F_0_899, F_1_175, F_1_501, -F_1_847
        -16069, 16819, -20995, 25172, // -F_1_961, F_2_053, -F_2_562, F_3_072
    ];

    /// libjpeg PASS1_BITS (intermediate precision carried across pass 1).
    const PASS1_BITS: i32 = 2;
    /// libjpeg DESCALE_P1 = CONST_BITS (13) − PASS1_BITS (2) = 11.
    /// Used for pass-1 multiplied-output descales.
    const DESCALE_P1: i32 = 11;
    /// libjpeg DESCALE_P2 = CONST_BITS (13) + PASS1_BITS (2) = 15.
    /// Used for pass-2 multiplied-output descales.
    const DESCALE_P2: i32 = 15;

    /// NEON forward DCT, in-place. Bit-exact equivalent to
    /// `fdct_kernel_scalar`. See module docs for the algorithm spec.
    ///
    /// # Safety
    ///
    /// `target_arch = "aarch64"` guarantees NEON; this is the only
    /// caller-visible safety requirement and it is enforced by the
    /// `#[cfg(target_arch = "aarch64")]` gate on the surrounding
    /// module. `data` is a fixed-size mutable reference; no further
    /// caller-side invariants beyond standard borrow rules.
    #[target_feature(enable = "neon")]
    pub(in crate::codec::jpeg) unsafe fn fdct_kernel_neon(data: &mut [i16; 64]) {
        let consts1 = vld1_s16(FDCT_CONSTS.as_ptr());
        let consts2 = vld1_s16(FDCT_CONSTS.as_ptr().add(4));
        let consts3 = vld1_s16(FDCT_CONSTS.as_ptr().add(8));

        // ---- Load 8 rows, transpose so each register holds one column. ----
        // `vld4q_s16` does a 4-way de-interleaved load — read 32 i16, but
        // the four output registers each get every-4th lane. Doing this
        // on rows 0..3 and rows 4..7 then merging with `vuzpq_s16` lands
        // each register on a single column.
        let s_rows_0123 = vld4q_s16(data.as_ptr());
        let s_rows_4567 = vld4q_s16(data.as_ptr().add(4 * 8));

        let cols_04 = vuzpq_s16(s_rows_0123.0, s_rows_4567.0);
        let cols_15 = vuzpq_s16(s_rows_0123.1, s_rows_4567.1);
        let cols_26 = vuzpq_s16(s_rows_0123.2, s_rows_4567.2);
        let cols_37 = vuzpq_s16(s_rows_0123.3, s_rows_4567.3);

        let mut col0 = cols_04.0;
        let mut col1 = cols_15.0;
        let mut col2 = cols_26.0;
        let mut col3 = cols_37.0;
        let mut col4 = cols_04.1;
        let mut col5 = cols_15.1;
        let mut col6 = cols_26.1;
        let mut col7 = cols_37.1;

        // ============================================================
        // Pass 1 — rows. (Registers currently hold *columns*; the 1D
        // butterfly is applied lane-wise across all 8 columns at once.)
        // ============================================================

        // Butterfly stage 1
        let tmp0 = vaddq_s16(col0, col7);
        let tmp7 = vsubq_s16(col0, col7);
        let tmp1 = vaddq_s16(col1, col6);
        let tmp6 = vsubq_s16(col1, col6);
        let tmp2 = vaddq_s16(col2, col5);
        let tmp5 = vsubq_s16(col2, col5);
        let tmp3 = vaddq_s16(col3, col4);
        let tmp4 = vsubq_s16(col3, col4);

        // Even-part butterfly
        let tmp10 = vaddq_s16(tmp0, tmp3);
        let tmp13 = vsubq_s16(tmp0, tmp3);
        let tmp11 = vaddq_s16(tmp1, tmp2);
        let tmp12 = vsubq_s16(tmp1, tmp2);

        // Outputs 0 and 4 — no multiply; just left-shift by PASS1_BITS.
        col0 = vshlq_n_s16(vaddq_s16(tmp10, tmp11), PASS1_BITS);
        col4 = vshlq_n_s16(vsubq_s16(tmp10, tmp11), PASS1_BITS);

        // Outputs 2 and 6 — even-part multiply + accumulate + descale.
        let tmp12_add_tmp13 = vaddq_s16(tmp12, tmp13);
        let z1_l = vmull_lane_s16::<2>(vget_low_s16(tmp12_add_tmp13), consts1); // *F_0_541
        let z1_h = vmull_lane_s16::<2>(vget_high_s16(tmp12_add_tmp13), consts1);

        let col2_l = vmlal_lane_s16::<3>(z1_l, vget_low_s16(tmp13), consts1); // +tmp13 *F_0_765
        let col2_h = vmlal_lane_s16::<3>(z1_h, vget_high_s16(tmp13), consts1);
        col2 = vcombine_s16(
            vrshrn_n_s32::<DESCALE_P1>(col2_l),
            vrshrn_n_s32::<DESCALE_P1>(col2_h),
        );

        let col6_l = vmlal_lane_s16::<3>(z1_l, vget_low_s16(tmp12), consts2); // +tmp12 *(-F_1_847)
        let col6_h = vmlal_lane_s16::<3>(z1_h, vget_high_s16(tmp12), consts2);
        col6 = vcombine_s16(
            vrshrn_n_s32::<DESCALE_P1>(col6_l),
            vrshrn_n_s32::<DESCALE_P1>(col6_h),
        );

        // Odd-part pre-sums
        let z1 = vaddq_s16(tmp4, tmp7);
        let z2 = vaddq_s16(tmp5, tmp6);
        let z3 = vaddq_s16(tmp4, tmp6);
        let z4 = vaddq_s16(tmp5, tmp7);

        // z5 = (z3 + z4) * F_1_175
        let mut z5_l = vmull_lane_s16::<1>(vget_low_s16(z3), consts2);
        let mut z5_h = vmull_lane_s16::<1>(vget_high_s16(z3), consts2);
        z5_l = vmlal_lane_s16::<1>(z5_l, vget_low_s16(z4), consts2);
        z5_h = vmlal_lane_s16::<1>(z5_h, vget_high_s16(z4), consts2);

        // Per-tmp multiplies
        let mut tmp4_l = vmull_lane_s16::<0>(vget_low_s16(tmp4), consts1); // tmp4 *F_0_298
        let mut tmp4_h = vmull_lane_s16::<0>(vget_high_s16(tmp4), consts1);
        let mut tmp5_l = vmull_lane_s16::<1>(vget_low_s16(tmp5), consts3); // tmp5 *F_2_053
        let mut tmp5_h = vmull_lane_s16::<1>(vget_high_s16(tmp5), consts3);
        let mut tmp6_l = vmull_lane_s16::<3>(vget_low_s16(tmp6), consts3); // tmp6 *F_3_072
        let mut tmp6_h = vmull_lane_s16::<3>(vget_high_s16(tmp6), consts3);
        let mut tmp7_l = vmull_lane_s16::<2>(vget_low_s16(tmp7), consts2); // tmp7 *F_1_501
        let mut tmp7_h = vmull_lane_s16::<2>(vget_high_s16(tmp7), consts2);

        // Per-z multiplies (z1..z4) — these are negated FIX_* constants
        // in the table, so the multiply naturally has the sign.
        let z1_l = vmull_lane_s16::<0>(vget_low_s16(z1), consts2); // z1 *(-F_0_899)
        let z1_h = vmull_lane_s16::<0>(vget_high_s16(z1), consts2);
        let z2_l = vmull_lane_s16::<2>(vget_low_s16(z2), consts3); // z2 *(-F_2_562)
        let z2_h = vmull_lane_s16::<2>(vget_high_s16(z2), consts3);
        let mut z3_l = vmull_lane_s16::<0>(vget_low_s16(z3), consts3); // z3 *(-F_1_961)
        let mut z3_h = vmull_lane_s16::<0>(vget_high_s16(z3), consts3);
        let mut z4_l = vmull_lane_s16::<1>(vget_low_s16(z4), consts1); // z4 *(-F_0_390)
        let mut z4_h = vmull_lane_s16::<1>(vget_high_s16(z4), consts1);

        z3_l = vaddq_s32(z3_l, z5_l);
        z3_h = vaddq_s32(z3_h, z5_h);
        z4_l = vaddq_s32(z4_l, z5_l);
        z4_h = vaddq_s32(z4_h, z5_h);

        // Output 7 = descale(tmp4 + z1 + z3)
        tmp4_l = vaddq_s32(tmp4_l, z1_l);
        tmp4_h = vaddq_s32(tmp4_h, z1_h);
        tmp4_l = vaddq_s32(tmp4_l, z3_l);
        tmp4_h = vaddq_s32(tmp4_h, z3_h);
        col7 = vcombine_s16(
            vrshrn_n_s32::<DESCALE_P1>(tmp4_l),
            vrshrn_n_s32::<DESCALE_P1>(tmp4_h),
        );

        // Output 5 = descale(tmp5 + z2 + z4)
        tmp5_l = vaddq_s32(tmp5_l, z2_l);
        tmp5_h = vaddq_s32(tmp5_h, z2_h);
        tmp5_l = vaddq_s32(tmp5_l, z4_l);
        tmp5_h = vaddq_s32(tmp5_h, z4_h);
        col5 = vcombine_s16(
            vrshrn_n_s32::<DESCALE_P1>(tmp5_l),
            vrshrn_n_s32::<DESCALE_P1>(tmp5_h),
        );

        // Output 3 = descale(tmp6 + z2 + z3)
        tmp6_l = vaddq_s32(tmp6_l, z2_l);
        tmp6_h = vaddq_s32(tmp6_h, z2_h);
        tmp6_l = vaddq_s32(tmp6_l, z3_l);
        tmp6_h = vaddq_s32(tmp6_h, z3_h);
        col3 = vcombine_s16(
            vrshrn_n_s32::<DESCALE_P1>(tmp6_l),
            vrshrn_n_s32::<DESCALE_P1>(tmp6_h),
        );

        // Output 1 = descale(tmp7 + z1 + z4)
        tmp7_l = vaddq_s32(tmp7_l, z1_l);
        tmp7_h = vaddq_s32(tmp7_h, z1_h);
        tmp7_l = vaddq_s32(tmp7_l, z4_l);
        tmp7_h = vaddq_s32(tmp7_h, z4_h);
        col1 = vcombine_s16(
            vrshrn_n_s32::<DESCALE_P1>(tmp7_l),
            vrshrn_n_s32::<DESCALE_P1>(tmp7_h),
        );

        // ---- Transpose: registers now hold *rows*. ----
        let cols_01 = vtrnq_s16(col0, col1);
        let cols_23 = vtrnq_s16(col2, col3);
        let cols_45 = vtrnq_s16(col4, col5);
        let cols_67 = vtrnq_s16(col6, col7);

        let cols_0145_l = vtrnq_s32(
            vreinterpretq_s32_s16(cols_01.0),
            vreinterpretq_s32_s16(cols_45.0),
        );
        let cols_0145_h = vtrnq_s32(
            vreinterpretq_s32_s16(cols_01.1),
            vreinterpretq_s32_s16(cols_45.1),
        );
        let cols_2367_l = vtrnq_s32(
            vreinterpretq_s32_s16(cols_23.0),
            vreinterpretq_s32_s16(cols_67.0),
        );
        let cols_2367_h = vtrnq_s32(
            vreinterpretq_s32_s16(cols_23.1),
            vreinterpretq_s32_s16(cols_67.1),
        );

        let rows_04 = vzipq_s32(cols_0145_l.0, cols_2367_l.0);
        let rows_15 = vzipq_s32(cols_0145_h.0, cols_2367_h.0);
        let rows_26 = vzipq_s32(cols_0145_l.1, cols_2367_l.1);
        let rows_37 = vzipq_s32(cols_0145_h.1, cols_2367_h.1);

        let mut row0 = vreinterpretq_s16_s32(rows_04.0);
        let mut row1 = vreinterpretq_s16_s32(rows_15.0);
        let mut row2 = vreinterpretq_s16_s32(rows_26.0);
        let mut row3 = vreinterpretq_s16_s32(rows_37.0);
        let mut row4 = vreinterpretq_s16_s32(rows_04.1);
        let mut row5 = vreinterpretq_s16_s32(rows_15.1);
        let mut row6 = vreinterpretq_s16_s32(rows_26.1);
        let mut row7 = vreinterpretq_s16_s32(rows_37.1);

        // ============================================================
        // Pass 2 — columns. (Same 1D butterfly, but applied to rows
        // and using DESCALE_P2 for the multiplied outputs to remove the
        // pass-1 promotion + constant-multiply scale.)
        // ============================================================

        let tmp0 = vaddq_s16(row0, row7);
        let tmp7 = vsubq_s16(row0, row7);
        let tmp1 = vaddq_s16(row1, row6);
        let tmp6 = vsubq_s16(row1, row6);
        let tmp2 = vaddq_s16(row2, row5);
        let tmp5 = vsubq_s16(row2, row5);
        let tmp3 = vaddq_s16(row3, row4);
        let tmp4 = vsubq_s16(row3, row4);

        let tmp10 = vaddq_s16(tmp0, tmp3);
        let tmp13 = vsubq_s16(tmp0, tmp3);
        let tmp11 = vaddq_s16(tmp1, tmp2);
        let tmp12 = vsubq_s16(tmp1, tmp2);

        // Outputs 0 and 4 — pass-2 rounding-shift-right by PASS1_BITS.
        row0 = vrshrq_n_s16::<PASS1_BITS>(vaddq_s16(tmp10, tmp11));
        row4 = vrshrq_n_s16::<PASS1_BITS>(vsubq_s16(tmp10, tmp11));

        let tmp12_add_tmp13 = vaddq_s16(tmp12, tmp13);
        let z1_l = vmull_lane_s16::<2>(vget_low_s16(tmp12_add_tmp13), consts1);
        let z1_h = vmull_lane_s16::<2>(vget_high_s16(tmp12_add_tmp13), consts1);

        let row2_l = vmlal_lane_s16::<3>(z1_l, vget_low_s16(tmp13), consts1);
        let row2_h = vmlal_lane_s16::<3>(z1_h, vget_high_s16(tmp13), consts1);
        row2 = vcombine_s16(
            vrshrn_n_s32::<DESCALE_P2>(row2_l),
            vrshrn_n_s32::<DESCALE_P2>(row2_h),
        );

        let row6_l = vmlal_lane_s16::<3>(z1_l, vget_low_s16(tmp12), consts2);
        let row6_h = vmlal_lane_s16::<3>(z1_h, vget_high_s16(tmp12), consts2);
        row6 = vcombine_s16(
            vrshrn_n_s32::<DESCALE_P2>(row6_l),
            vrshrn_n_s32::<DESCALE_P2>(row6_h),
        );

        let z1 = vaddq_s16(tmp4, tmp7);
        let z2 = vaddq_s16(tmp5, tmp6);
        let z3 = vaddq_s16(tmp4, tmp6);
        let z4 = vaddq_s16(tmp5, tmp7);

        let mut z5_l = vmull_lane_s16::<1>(vget_low_s16(z3), consts2);
        let mut z5_h = vmull_lane_s16::<1>(vget_high_s16(z3), consts2);
        z5_l = vmlal_lane_s16::<1>(z5_l, vget_low_s16(z4), consts2);
        z5_h = vmlal_lane_s16::<1>(z5_h, vget_high_s16(z4), consts2);

        let mut tmp4_l = vmull_lane_s16::<0>(vget_low_s16(tmp4), consts1);
        let mut tmp4_h = vmull_lane_s16::<0>(vget_high_s16(tmp4), consts1);
        let mut tmp5_l = vmull_lane_s16::<1>(vget_low_s16(tmp5), consts3);
        let mut tmp5_h = vmull_lane_s16::<1>(vget_high_s16(tmp5), consts3);
        let mut tmp6_l = vmull_lane_s16::<3>(vget_low_s16(tmp6), consts3);
        let mut tmp6_h = vmull_lane_s16::<3>(vget_high_s16(tmp6), consts3);
        let mut tmp7_l = vmull_lane_s16::<2>(vget_low_s16(tmp7), consts2);
        let mut tmp7_h = vmull_lane_s16::<2>(vget_high_s16(tmp7), consts2);

        let z1_l = vmull_lane_s16::<0>(vget_low_s16(z1), consts2);
        let z1_h = vmull_lane_s16::<0>(vget_high_s16(z1), consts2);
        let z2_l = vmull_lane_s16::<2>(vget_low_s16(z2), consts3);
        let z2_h = vmull_lane_s16::<2>(vget_high_s16(z2), consts3);
        let mut z3_l = vmull_lane_s16::<0>(vget_low_s16(z3), consts3);
        let mut z3_h = vmull_lane_s16::<0>(vget_high_s16(z3), consts3);
        let mut z4_l = vmull_lane_s16::<1>(vget_low_s16(z4), consts1);
        let mut z4_h = vmull_lane_s16::<1>(vget_high_s16(z4), consts1);

        z3_l = vaddq_s32(z3_l, z5_l);
        z3_h = vaddq_s32(z3_h, z5_h);
        z4_l = vaddq_s32(z4_l, z5_l);
        z4_h = vaddq_s32(z4_h, z5_h);

        tmp4_l = vaddq_s32(tmp4_l, z1_l);
        tmp4_h = vaddq_s32(tmp4_h, z1_h);
        tmp4_l = vaddq_s32(tmp4_l, z3_l);
        tmp4_h = vaddq_s32(tmp4_h, z3_h);
        row7 = vcombine_s16(
            vrshrn_n_s32::<DESCALE_P2>(tmp4_l),
            vrshrn_n_s32::<DESCALE_P2>(tmp4_h),
        );

        tmp5_l = vaddq_s32(tmp5_l, z2_l);
        tmp5_h = vaddq_s32(tmp5_h, z2_h);
        tmp5_l = vaddq_s32(tmp5_l, z4_l);
        tmp5_h = vaddq_s32(tmp5_h, z4_h);
        row5 = vcombine_s16(
            vrshrn_n_s32::<DESCALE_P2>(tmp5_l),
            vrshrn_n_s32::<DESCALE_P2>(tmp5_h),
        );

        tmp6_l = vaddq_s32(tmp6_l, z2_l);
        tmp6_h = vaddq_s32(tmp6_h, z2_h);
        tmp6_l = vaddq_s32(tmp6_l, z3_l);
        tmp6_h = vaddq_s32(tmp6_h, z3_h);
        row3 = vcombine_s16(
            vrshrn_n_s32::<DESCALE_P2>(tmp6_l),
            vrshrn_n_s32::<DESCALE_P2>(tmp6_h),
        );

        tmp7_l = vaddq_s32(tmp7_l, z1_l);
        tmp7_h = vaddq_s32(tmp7_h, z1_h);
        tmp7_l = vaddq_s32(tmp7_l, z4_l);
        tmp7_h = vaddq_s32(tmp7_h, z4_h);
        row1 = vcombine_s16(
            vrshrn_n_s32::<DESCALE_P2>(tmp7_l),
            vrshrn_n_s32::<DESCALE_P2>(tmp7_h),
        );

        // ---- Store rows. ----
        let p = data.as_mut_ptr();
        vst1q_s16(p, row0);
        vst1q_s16(p.add(8), row1);
        vst1q_s16(p.add(16), row2);
        vst1q_s16(p.add(24), row3);
        vst1q_s16(p.add(32), row4);
        vst1q_s16(p.add(40), row5);
        vst1q_s16(p.add(48), row6);
        vst1q_s16(p.add(56), row7);
    }

    // ====================================================================
    // IDCT — Inverse DCT, 4-wide
    // ====================================================================
    //
    // Direct port of libjpeg-turbo `simd/arm/jidctint-neon.c`
    // (Copyright (C) 2020, Arm Limited, zlib-style — the file's own
    // license clause permits free use, modification and redistribution
    // with the source-attribution requirement at the top of THIS file).
    //
    // The upstream uses an algebraic refactoring of the canonical
    // `jpeg_idct_islow` algorithm that combines multiplies into shared
    // expressions. The refactoring is **bit-exact** with the canonical
    // form in Q13 integer arithmetic. Phasm's
    // [`crate::codec::jpeg::pixels_aan::idct_kernel_scalar`] uses the
    // canonical (un-refactored) form; this NEON kernel uses the
    // refactored form. The NEON-vs-scalar bit-exact test in
    // `pixels_aan::tests::idct_neon_bitexact_vs_scalar` empirically
    // confirms equivalence on 500 random blocks + edge fixtures.
    //
    // Structural choices vs upstream:
    //
    // - **4-wide throughout** (matches upstream — 8-wide would spill).
    // - **Output is i32** (not u8 + clamp like upstream). The phasm
    //   wrapper adds +128 and converts to f64; clamp is the caller's.
    // - **No sparse-path optimizations** (single regular path).

    /// Q13 IDCT constants. Layout matches libjpeg-turbo
    /// `jsimd_idct_islow_neon_consts` so lane indices below match the
    /// upstream source line-by-line.
    ///
    /// | Register | Lane 0          | Lane 1          | Lane 2              | Lane 3              |
    /// |----------|-----------------|-----------------|---------------------|---------------------|
    /// | consts1  | F_0_899         | F_0_541         | F_2_562             | F_0_298 - F_0_899   |
    /// | consts2  | F_1_501-F_0_899 | F_2_053-F_2_562 | F_0_541+F_0_765     | F_1_175             |
    /// | consts3  | F_1_175-F_0_390 | F_0_541-F_1_847 | F_3_072-F_2_562     | F_1_175-F_1_961     |
    static IDCT_CONSTS: [i16; 12] = [
        7373, 4433,   // F_0_899, F_0_541
        20995, -4927, // F_2_562, F_0_298 - F_0_899
        4926, -4176,  // F_1_501 - F_0_899, F_2_053 - F_2_562
        10703, 9633,  // F_0_541 + F_0_765, F_1_175
        6437, -10704, // F_1_175 - F_0_390, F_0_541 - F_1_847
        4177, -6436,  // F_3_072 - F_2_562, F_1_175 - F_1_961
    ];

    /// Pass-1 multiplied-output descale (CONST_BITS - PASS1_BITS = 11).
    const IDCT_DESCALE_P1: i32 = 11;
    /// Pass-2 multiplied-output descale (CONST_BITS + PASS1_BITS + 3 = 18).
    const IDCT_DESCALE_P2: i32 = 18;

    /// NEON inverse DCT.
    ///
    /// # Safety
    /// `target_arch = "aarch64"` guarantees NEON.
    #[target_feature(enable = "neon")]
    pub(in crate::codec::jpeg) unsafe fn idct_kernel_neon(
        quantized: &[i16; 64],
        qt: &[u16; 64],
        out: &mut [i32; 64],
    ) {
        // JPEG-spec QT values are ≤ 255 for 8-bit baseline JPEG.
        // Reinterpret u16 as i16 — values stay non-negative.
        let qt_ptr = qt.as_ptr() as *const i16;

        // i16 workspace: 32 elements per half.
        //   ws_l = post-pass-1 output for rows 0..3 (all 8 cols).
        //   ws_r = post-pass-1 output for rows 4..7 (all 8 cols).
        // VST4 stores in pass 1 transpose each 4×4 quadrant.
        let mut ws_l = [0i16; 32];
        let mut ws_r = [0i16; 32];

        // Pass 1 left half (input cols 0..3) → ws_l[0..16] + ws_r[0..16].
        idct_pass1_4cols(quantized.as_ptr(), qt_ptr, 0, ws_l.as_mut_ptr(), ws_r.as_mut_ptr());
        // Pass 1 right half (input cols 4..7) → ws_l[16..32] + ws_r[16..32].
        idct_pass1_4cols(
            quantized.as_ptr(),
            qt_ptr,
            4,
            ws_l.as_mut_ptr().add(16),
            ws_r.as_mut_ptr().add(16),
        );

        // Pass 2 top (ws_l → out rows 0..3) + bottom (ws_r → out rows 4..7).
        idct_pass2_4rows(ws_l.as_ptr(), out.as_mut_ptr(), 0);
        idct_pass2_4rows(ws_r.as_ptr(), out.as_mut_ptr(), 4);
    }

    /// IDCT pass 1, processing a 4-col-wide slice of the input.
    ///
    /// # Safety
    /// All pointers valid for the documented access ranges.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn idct_pass1_4cols(
        coef: *const i16,
        qt: *const i16,
        col_off: usize,
        ws_top: *mut i16,
        ws_bot: *mut i16,
    ) {
        let consts1 = vld1_s16(IDCT_CONSTS.as_ptr());
        let consts2 = vld1_s16(IDCT_CONSTS.as_ptr().add(4));
        let consts3 = vld1_s16(IDCT_CONSTS.as_ptr().add(8));

        let row0 = vld1_s16(coef.add(0 * 8 + col_off));
        let row1 = vld1_s16(coef.add(1 * 8 + col_off));
        let row2 = vld1_s16(coef.add(2 * 8 + col_off));
        let row3 = vld1_s16(coef.add(3 * 8 + col_off));
        let row4 = vld1_s16(coef.add(4 * 8 + col_off));
        let row5 = vld1_s16(coef.add(5 * 8 + col_off));
        let row6 = vld1_s16(coef.add(6 * 8 + col_off));
        let row7 = vld1_s16(coef.add(7 * 8 + col_off));

        let q0 = vld1_s16(qt.add(0 * 8 + col_off));
        let q1 = vld1_s16(qt.add(1 * 8 + col_off));
        let q2 = vld1_s16(qt.add(2 * 8 + col_off));
        let q3 = vld1_s16(qt.add(3 * 8 + col_off));
        let q4 = vld1_s16(qt.add(4 * 8 + col_off));
        let q5 = vld1_s16(qt.add(5 * 8 + col_off));
        let q6 = vld1_s16(qt.add(6 * 8 + col_off));
        let q7 = vld1_s16(qt.add(7 * 8 + col_off));

        // ---- Even part (refactored from canonical). ----
        let z2 = vmul_s16(row2, q2);
        let z3 = vmul_s16(row6, q6);
        let mut tmp2 = vmull_lane_s16::<1>(z2, consts1); // z2 * F_0_541
        let mut tmp3 = vmull_lane_s16::<2>(z2, consts2); // z2 * (F_0_541+F_0_765)
        tmp2 = vmlal_lane_s16::<1>(tmp2, z3, consts3); // += z3 * (F_0_541-F_1_847)
        tmp3 = vmlal_lane_s16::<1>(tmp3, z3, consts1); // += z3 * F_0_541

        let z2 = vmul_s16(row0, q0);
        let z3 = vmul_s16(row4, q4);
        let tmp0 = vshll_n_s16::<13>(vadd_s16(z2, z3));
        let tmp1 = vshll_n_s16::<13>(vsub_s16(z2, z3));

        let tmp10 = vaddq_s32(tmp0, tmp3);
        let tmp13 = vsubq_s32(tmp0, tmp3);
        let tmp11 = vaddq_s32(tmp1, tmp2);
        let tmp12 = vsubq_s32(tmp1, tmp2);

        // ---- Odd part. ----
        let t0 = vmul_s16(row7, q7);
        let t1 = vmul_s16(row5, q5);
        let t2 = vmul_s16(row3, q3);
        let t3 = vmul_s16(row1, q1);

        let z3_s16 = vadd_s16(t0, t2);
        let z4_s16 = vadd_s16(t1, t3);

        let mut z3 = vmull_lane_s16::<3>(z3_s16, consts3);
        let mut z4 = vmull_lane_s16::<3>(z3_s16, consts2);
        z3 = vmlal_lane_s16::<3>(z3, z4_s16, consts2);
        z4 = vmlal_lane_s16::<0>(z4, z4_s16, consts3);

        let mut tmp0 = vmull_lane_s16::<3>(t0, consts1);
        let mut tmp1 = vmull_lane_s16::<1>(t1, consts2);
        let mut tmp2 = vmull_lane_s16::<2>(t2, consts3);
        let mut tmp3 = vmull_lane_s16::<0>(t3, consts2);

        tmp0 = vmlsl_lane_s16::<0>(tmp0, t3, consts1);
        tmp1 = vmlsl_lane_s16::<2>(tmp1, t2, consts1);
        tmp2 = vmlsl_lane_s16::<2>(tmp2, t1, consts1);
        tmp3 = vmlsl_lane_s16::<0>(tmp3, t0, consts1);

        tmp0 = vaddq_s32(tmp0, z3);
        tmp1 = vaddq_s32(tmp1, z4);
        tmp2 = vaddq_s32(tmp2, z3);
        tmp3 = vaddq_s32(tmp3, z4);

        // ---- Descale + narrow to i16, output 8 rows via two VST4 transposes. ----
        let r0 = vrshrn_n_s32::<IDCT_DESCALE_P1>(vaddq_s32(tmp10, tmp3));
        let r1 = vrshrn_n_s32::<IDCT_DESCALE_P1>(vaddq_s32(tmp11, tmp2));
        let r2 = vrshrn_n_s32::<IDCT_DESCALE_P1>(vaddq_s32(tmp12, tmp1));
        let r3 = vrshrn_n_s32::<IDCT_DESCALE_P1>(vaddq_s32(tmp13, tmp0));
        let r4 = vrshrn_n_s32::<IDCT_DESCALE_P1>(vsubq_s32(tmp13, tmp0));
        let r5 = vrshrn_n_s32::<IDCT_DESCALE_P1>(vsubq_s32(tmp12, tmp1));
        let r6 = vrshrn_n_s32::<IDCT_DESCALE_P1>(vsubq_s32(tmp11, tmp2));
        let r7 = vrshrn_n_s32::<IDCT_DESCALE_P1>(vsubq_s32(tmp10, tmp3));

        vst4_s16(ws_top, int16x4x4_t(r0, r1, r2, r3));
        vst4_s16(ws_bot, int16x4x4_t(r4, r5, r6, r7));
    }

    /// IDCT pass 2, processing 4 output rows.
    ///
    /// # Safety
    /// All pointers valid for documented ranges.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn idct_pass2_4rows(ws: *const i16, out: *mut i32, row_off: usize) {
        let consts1 = vld1_s16(IDCT_CONSTS.as_ptr());
        let consts2 = vld1_s16(IDCT_CONSTS.as_ptr().add(4));
        let consts3 = vld1_s16(IDCT_CONSTS.as_ptr().add(8));

        // Each loaded vector = 4 rows (lanes 0..3) of one col of post-pass-1.
        let col0 = vld1_s16(ws.add(0 * 4));
        let col1 = vld1_s16(ws.add(1 * 4));
        let col2 = vld1_s16(ws.add(2 * 4));
        let col3 = vld1_s16(ws.add(3 * 4));
        let col4 = vld1_s16(ws.add(4 * 4));
        let col5 = vld1_s16(ws.add(5 * 4));
        let col6 = vld1_s16(ws.add(6 * 4));
        let col7 = vld1_s16(ws.add(7 * 4));

        // ---- Even part. ----
        let mut tmp2 = vmull_lane_s16::<1>(col2, consts1);
        let mut tmp3 = vmull_lane_s16::<2>(col2, consts2);
        tmp2 = vmlal_lane_s16::<1>(tmp2, col6, consts3);
        tmp3 = vmlal_lane_s16::<1>(tmp3, col6, consts1);

        let tmp0 = vshll_n_s16::<13>(vadd_s16(col0, col4));
        let tmp1 = vshll_n_s16::<13>(vsub_s16(col0, col4));

        let tmp10 = vaddq_s32(tmp0, tmp3);
        let tmp13 = vsubq_s32(tmp0, tmp3);
        let tmp11 = vaddq_s32(tmp1, tmp2);
        let tmp12 = vsubq_s32(tmp1, tmp2);

        // ---- Odd part. ----
        let z3_s16 = vadd_s16(col7, col3);
        let z4_s16 = vadd_s16(col5, col1);

        let mut z3 = vmull_lane_s16::<3>(z3_s16, consts3);
        let mut z4 = vmull_lane_s16::<3>(z3_s16, consts2);
        z3 = vmlal_lane_s16::<3>(z3, z4_s16, consts2);
        z4 = vmlal_lane_s16::<0>(z4, z4_s16, consts3);

        let mut tmp0 = vmull_lane_s16::<3>(col7, consts1);
        let mut tmp1 = vmull_lane_s16::<1>(col5, consts2);
        let mut tmp2 = vmull_lane_s16::<2>(col3, consts3);
        let mut tmp3 = vmull_lane_s16::<0>(col1, consts2);

        tmp0 = vmlsl_lane_s16::<0>(tmp0, col1, consts1);
        tmp1 = vmlsl_lane_s16::<2>(tmp1, col3, consts1);
        tmp2 = vmlsl_lane_s16::<2>(tmp2, col5, consts1);
        tmp3 = vmlsl_lane_s16::<0>(tmp3, col7, consts1);

        tmp0 = vaddq_s32(tmp0, z3);
        tmp1 = vaddq_s32(tmp1, z4);
        tmp2 = vaddq_s32(tmp2, z3);
        tmp3 = vaddq_s32(tmp3, z4);

        // ---- Final descale by 18, keep as i32. ----
        let oc0 = vrshrq_n_s32::<IDCT_DESCALE_P2>(vaddq_s32(tmp10, tmp3));
        let oc1 = vrshrq_n_s32::<IDCT_DESCALE_P2>(vaddq_s32(tmp11, tmp2));
        let oc2 = vrshrq_n_s32::<IDCT_DESCALE_P2>(vaddq_s32(tmp12, tmp1));
        let oc3 = vrshrq_n_s32::<IDCT_DESCALE_P2>(vaddq_s32(tmp13, tmp0));
        let oc4 = vrshrq_n_s32::<IDCT_DESCALE_P2>(vsubq_s32(tmp13, tmp0));
        let oc5 = vrshrq_n_s32::<IDCT_DESCALE_P2>(vsubq_s32(tmp12, tmp1));
        let oc6 = vrshrq_n_s32::<IDCT_DESCALE_P2>(vsubq_s32(tmp11, tmp2));
        let oc7 = vrshrq_n_s32::<IDCT_DESCALE_P2>(vsubq_s32(tmp10, tmp3));

        // Each oc_N is i32x4 with lane k = output for (row=k, col=N).
        // Transpose to row-major and store.
        let (or0_lo, or1_lo, or2_lo, or3_lo) = transpose_4x4_i32(oc0, oc1, oc2, oc3);
        let (or0_hi, or1_hi, or2_hi, or3_hi) = transpose_4x4_i32(oc4, oc5, oc6, oc7);

        vst1q_s32(out.add((row_off + 0) * 8), or0_lo);
        vst1q_s32(out.add((row_off + 0) * 8 + 4), or0_hi);
        vst1q_s32(out.add((row_off + 1) * 8), or1_lo);
        vst1q_s32(out.add((row_off + 1) * 8 + 4), or1_hi);
        vst1q_s32(out.add((row_off + 2) * 8), or2_lo);
        vst1q_s32(out.add((row_off + 2) * 8 + 4), or2_hi);
        vst1q_s32(out.add((row_off + 3) * 8), or3_lo);
        vst1q_s32(out.add((row_off + 3) * 8 + 4), or3_hi);
    }

    /// 4×4 i32 transpose. cN[r] = value for (row r, col N) →
    /// outputs rN[c] = value for (row N, col c).
    ///
    /// # Safety
    /// NEON via cfg gate.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn transpose_4x4_i32(
        c0: int32x4_t,
        c1: int32x4_t,
        c2: int32x4_t,
        c3: int32x4_t,
    ) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
        // vtrnq_s32: (.0) = [c0r0, c1r0, c0r2, c1r2], (.1) = [c0r1, c1r1, c0r3, c1r3]
        let t01 = vtrnq_s32(c0, c1);
        let t23 = vtrnq_s32(c2, c3);
        // Combine low halves → rows 0,1 ; high halves → rows 2,3.
        let r0 = vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0));
        let r1 = vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1));
        let r2 = vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0));
        let r3 = vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1));
        (r0, r1, r2, r3)
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) mod avx2 {
    //! AVX2 LL&M DCT/IDCT (x86_64).
    //!
    //! The fdct is a direct port of `naoto256/jpeg-rusturbo` v0.3.0
    //! `src/arch/x86_64.rs::fdct_avx2` (MIT/Apache-2.0), itself a Rust
    //! translation of libjpeg-turbo `simd/x86_64/jfdctint-avx2.asm`
    //! (BSD-3-Clause + IJG). The IDCT (below) was written directly from
    //! `simd/x86_64/jidctint-avx2.asm` since no Rust reference exists.
    //!
    //! Algorithmic choices preserved from upstream:
    //!
    //! - Two rows per `__m256i` (16 i16 lanes = lo-128 holds row N,
    //!   hi-128 holds row N+4). The whole 8×8 fits in 4 ymm.
    //! - `_mm256_madd_epi16` for pair-sum LL&M multiplies. The 8 packed
    //!   constants tables (`PW_F130_F054…` etc.) are pre-computed
    //!   pair-sums of the Q13 FIX_* values; verified bit-for-bit
    //!   against the canonical algorithm.
    //! - `dotranspose` does a 4-ymm 8×8 transpose via
    //!   `unpacklo/hi_epi16/32` + `permute4x64`.
    //! - `dodct<const PASS: i32>` const-generic; the only pass-1/pass-2
    //!   difference is descale shift (11 vs 15) + the pre-shift in
    //!   pass 2 for the DC outputs.
    //! - `_mm256_zeroupper()` at exit avoids the SSE/AVX transition
    //!   penalty on older Intel.

    use core::arch::x86_64::*;

    #[repr(C, align(32))]
    struct Aligned32<T>(T);

    /// PW_F130_F054_MF130_F054 — pair (F_0_541 + F_0_765, F_0_541) ×4
    /// then pair (F_0_541 - F_1_847, F_0_541) ×4.
    static PW_F130_F054_MF130_F054: Aligned32<[i16; 16]> = Aligned32([
        10703, 4433, 10703, 4433, 10703, 4433, 10703, 4433, -10704, 4433, -10704, 4433, -10704,
        4433, -10704, 4433,
    ]);

    /// PW_MF078_F117_F078_F117.
    static PW_MF078_F117_F078_F117: Aligned32<[i16; 16]> = Aligned32([
        -6436, 9633, -6436, 9633, -6436, 9633, -6436, 9633, 6437, 9633, 6437, 9633, 6437, 9633,
        6437, 9633,
    ]);

    /// PW_MF060_MF089_MF050_MF256.
    static PW_MF060_MF089_MF050_MF256: Aligned32<[i16; 16]> = Aligned32([
        -4927, -7373, -4927, -7373, -4927, -7373, -4927, -7373, -4176, -20995, -4176, -20995,
        -4176, -20995, -4176, -20995,
    ]);

    /// PW_F050_MF256_F060_MF089.
    static PW_F050_MF256_F060_MF089: Aligned32<[i16; 16]> = Aligned32([
        4177, -20995, 4177, -20995, 4177, -20995, 4177, -20995, 4926, -7373, 4926, -7373, 4926,
        -7373, 4926, -7373,
    ]);

    /// Descale round-bias for pass 1 (1 << (11-1) = 1024).
    static PD_DESCALE_P1: Aligned32<[i32; 8]> = Aligned32([1024; 8]);
    /// Descale round-bias for pass 2 (1 << (15-1) = 16384).
    static PD_DESCALE_P2: Aligned32<[i32; 8]> = Aligned32([16384; 8]);
    /// Pass-2 DC-output PASS1_BITS down-shift round-bias (1 << (2-1) = 2).
    static PW_DESCALE_P2X: Aligned32<[i16; 16]> = Aligned32([2; 16]);
    /// Lane-sign mask: low 128 = +1, high 128 = −1 (used by `vpsignw`
    /// to flip the sign of the upper half while leaving the lower
    /// half alone — encodes the packed `tmp10_neg11` swap).
    static PW_1_NEG1: Aligned32<[i16; 16]> = Aligned32([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]);

    #[inline(always)]
    unsafe fn load(p: *const i16) -> __m256i {
        unsafe { _mm256_loadu_si256(p as *const __m256i) }
    }

    /// 8×8×16-bit transpose. 4 input ymm each holding 2 rows packed
    /// → 4 output ymm each holding 2 columns packed.
    #[inline(always)]
    unsafe fn dotranspose(
        m1: __m256i,
        m2: __m256i,
        m3: __m256i,
        m4: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        unsafe {
            let t5 = _mm256_unpacklo_epi16(m1, m2);
            let t6 = _mm256_unpackhi_epi16(m1, m2);
            let t7 = _mm256_unpacklo_epi16(m3, m4);
            let t8 = _mm256_unpackhi_epi16(m3, m4);
            let m1 = _mm256_unpacklo_epi32(t5, t7);
            let m2 = _mm256_unpackhi_epi32(t5, t7);
            let m3 = _mm256_unpacklo_epi32(t6, t8);
            let m4 = _mm256_unpackhi_epi32(t6, t8);
            let m1 = _mm256_permute4x64_epi64::<0x8D>(m1);
            let m2 = _mm256_permute4x64_epi64::<0x8D>(m2);
            let m3 = _mm256_permute4x64_epi64::<0xD8>(m3);
            let m4 = _mm256_permute4x64_epi64::<0xD8>(m4);
            (m1, m2, m3, m4)
        }
    }

    /// 1D DCT pass over the 4 ymm. Const-generic on PASS (1 or 2)
    /// selects descale shift + the PASS1_BITS DC-output post-add only
    /// present in pass 2. Returns (data0_4, data3_1, data2_6, data7_5).
    #[inline(always)]
    unsafe fn dodct<const PASS: i32>(
        m1: __m256i,
        m2: __m256i,
        m3: __m256i,
        m4: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        unsafe {
            let m5 = _mm256_sub_epi16(m1, m4); // tmp6_7
            let m6 = _mm256_add_epi16(m1, m4); // tmp1_0
            let m7 = _mm256_add_epi16(m2, m3); // tmp3_2
            let m8 = _mm256_sub_epi16(m2, m3); // tmp4_5

            // -- Even
            let m6 = _mm256_permute2x128_si256::<0x01>(m6, m6); // tmp0_1
            let m1 = _mm256_add_epi16(m6, m7); // tmp10_11
            let m6 = _mm256_sub_epi16(m6, m7); // tmp13_12

            let m7 = _mm256_permute2x128_si256::<0x01>(m1, m1); // tmp11_10
            let pw_1_neg1 = load(PW_1_NEG1.0.as_ptr());
            let m1 = _mm256_sign_epi16(m1, pw_1_neg1); // tmp10_neg11
            let m7 = _mm256_add_epi16(m7, m1); // (tmp10+tmp11)_(tmp10-tmp11)

            let m1 = if PASS == 1 {
                _mm256_slli_epi16::<2>(m7) // data0_4 (up-shift by PASS1_BITS)
            } else {
                let pw_descale_p2x = load(PW_DESCALE_P2X.0.as_ptr());
                let m7 = _mm256_add_epi16(m7, pw_descale_p2x);
                _mm256_srai_epi16::<2>(m7) // data0_4 (down-shift by PASS1_BITS)
            };

            // -- data2_6 (even part continued)
            let m7 = _mm256_permute2x128_si256::<0x01>(m6, m6); // tmp12_13
            let m2_lo = _mm256_unpacklo_epi16(m6, m7);
            let m6_hi = _mm256_unpackhi_epi16(m6, m7);

            let pw_f130 = load(PW_F130_F054_MF130_F054.0.as_ptr());
            let m2_lo = _mm256_madd_epi16(m2_lo, pw_f130);
            let m6_hi = _mm256_madd_epi16(m6_hi, pw_f130);

            let pd_descale = if PASS == 1 {
                _mm256_loadu_si256(PD_DESCALE_P1.0.as_ptr() as *const __m256i)
            } else {
                _mm256_loadu_si256(PD_DESCALE_P2.0.as_ptr() as *const __m256i)
            };
            let m2_lo = _mm256_add_epi32(m2_lo, pd_descale);
            let m6_hi = _mm256_add_epi32(m6_hi, pd_descale);
            let (m2_lo, m6_hi) = if PASS == 1 {
                (_mm256_srai_epi32::<11>(m2_lo), _mm256_srai_epi32::<11>(m6_hi))
            } else {
                (_mm256_srai_epi32::<15>(m2_lo), _mm256_srai_epi32::<15>(m6_hi))
            };

            let m3 = _mm256_packs_epi32(m2_lo, m6_hi); // data2_6

            // -- Odd
            let m7 = _mm256_add_epi16(m8, m5); // z3_4

            let m2 = _mm256_permute2x128_si256::<0x01>(m7, m7); // z4_3
            let m6_lo = _mm256_unpacklo_epi16(m7, m2);
            let m7_hi = _mm256_unpackhi_epi16(m7, m2);

            let pw_mf078 = load(PW_MF078_F117_F078_F117.0.as_ptr());
            let m6_lo = _mm256_madd_epi16(m6_lo, pw_mf078); // z3_4 L
            let m7_hi = _mm256_madd_epi16(m7_hi, pw_mf078); // z3_4 H

            // -- data7_5
            let m4 = _mm256_permute2x128_si256::<0x01>(m5, m5); // tmp7_6
            let m2_lo = _mm256_unpacklo_epi16(m8, m4);
            let m4_hi = _mm256_unpackhi_epi16(m8, m4);

            let pw_mf060 = load(PW_MF060_MF089_MF050_MF256.0.as_ptr());
            let m2_lo = _mm256_madd_epi16(m2_lo, pw_mf060); // tmp4_5 L
            let m4_hi = _mm256_madd_epi16(m4_hi, pw_mf060); // tmp4_5 H

            let m2_lo = _mm256_add_epi32(m2_lo, m6_lo); // data7_5 L
            let m4_hi = _mm256_add_epi32(m4_hi, m7_hi); // data7_5 H

            let m2_lo = _mm256_add_epi32(m2_lo, pd_descale);
            let m4_hi = _mm256_add_epi32(m4_hi, pd_descale);
            let (m2_lo, m4_hi) = if PASS == 1 {
                (_mm256_srai_epi32::<11>(m2_lo), _mm256_srai_epi32::<11>(m4_hi))
            } else {
                (_mm256_srai_epi32::<15>(m2_lo), _mm256_srai_epi32::<15>(m4_hi))
            };

            let m4 = _mm256_packs_epi32(m2_lo, m4_hi); // data7_5

            // -- data3_1
            let m2 = _mm256_permute2x128_si256::<0x01>(m8, m8); // tmp5_4
            let m8_lo = _mm256_unpacklo_epi16(m5, m2);
            let m5_hi = _mm256_unpackhi_epi16(m5, m2);

            let pw_f050 = load(PW_F050_MF256_F060_MF089.0.as_ptr());
            let m8_lo = _mm256_madd_epi16(m8_lo, pw_f050); // tmp6_7 L
            let m5_hi = _mm256_madd_epi16(m5_hi, pw_f050); // tmp6_7 H

            let m8_lo = _mm256_add_epi32(m8_lo, m6_lo); // data3_1 L
            let m5_hi = _mm256_add_epi32(m5_hi, m7_hi); // data3_1 H

            let m8_lo = _mm256_add_epi32(m8_lo, pd_descale);
            let m5_hi = _mm256_add_epi32(m5_hi, pd_descale);
            let (m8_lo, m5_hi) = if PASS == 1 {
                (_mm256_srai_epi32::<11>(m8_lo), _mm256_srai_epi32::<11>(m5_hi))
            } else {
                (_mm256_srai_epi32::<15>(m8_lo), _mm256_srai_epi32::<15>(m5_hi))
            };

            let m2 = _mm256_packs_epi32(m8_lo, m5_hi); // data3_1

            (m1, m2, m3, m4)
        }
    }

    /// AVX2 forward DCT, in-place. Bit-exact equivalent to
    /// `fdct_kernel_scalar`.
    ///
    /// # Safety
    /// Caller must have AVX2 available (runtime check in
    /// `fdct_kernel`'s dispatcher).
    #[target_feature(enable = "avx2")]
    pub(in crate::codec::jpeg) unsafe fn fdct_kernel_avx2(data: &mut [i16; 64]) {
        let p = data.as_mut_ptr() as *mut __m256i;
        let m4 = _mm256_loadu_si256(p);
        let m5 = _mm256_loadu_si256(p.add(1));
        let m6 = _mm256_loadu_si256(p.add(2));
        let m7 = _mm256_loadu_si256(p.add(3));

        // Re-pack to (rows 0,4)(rows 1,5)(rows 2,6)(rows 3,7).
        let m0 = _mm256_permute2x128_si256::<0x20>(m4, m6);
        let m1 = _mm256_permute2x128_si256::<0x31>(m4, m6);
        let m2 = _mm256_permute2x128_si256::<0x20>(m5, m7);
        let m3 = _mm256_permute2x128_si256::<0x31>(m5, m7);

        // Pass 1.
        let (t0, t1, t2, t3) = dotranspose(m0, m1, m2, m3);
        let (out0, out1, out2, out3) = dodct::<1>(t0, t1, t2, t3);

        // Re-pack between passes.
        let p4 = _mm256_permute2x128_si256::<0x20>(out1, out3); // data3_7
        let p1 = _mm256_permute2x128_si256::<0x31>(out1, out3); // data1_5

        // Pass 2.
        let (t0, t1, t2, t3) = dotranspose(out0, p1, out2, p4);
        let (out0, out1, out2, out3) = dodct::<2>(t0, t1, t2, t3);

        // Re-pack into row order and store.
        let s0 = _mm256_permute2x128_si256::<0x30>(out0, out1);
        let s1 = _mm256_permute2x128_si256::<0x20>(out2, out1);
        let s2 = _mm256_permute2x128_si256::<0x31>(out0, out3);
        let s3 = _mm256_permute2x128_si256::<0x21>(out2, out3);

        _mm256_storeu_si256(p, s0);
        _mm256_storeu_si256(p.add(1), s1);
        _mm256_storeu_si256(p.add(2), s2);
        _mm256_storeu_si256(p.add(3), s3);

        _mm256_zeroupper();
    }

    // ====================================================================
    // IDCT — AVX2 inverse DCT
    // ====================================================================
    //
    // Direct port of libjpeg-turbo `simd/x86_64/jidctint-avx2.asm`
    // `DODCT` macro + `jsimd_idct_islow_avx2`. No Rust reference
    // existed (jpeg-rusturbo v0.3.0's AVX2 `idct_islow` falls back to
    // scalar, same as their NEON). Structurally mirrors the AVX2 fdct
    // above — same 2-rows-per-ymm packing, same `_mm256_madd_epi16`
    // pair-sum LL&M multiplies, same `dotranspose`. The only divergence
    // from upstream is the final output stage: upstream packs i16 → u8
    // with `+128` clamp; phasm keeps i32 to match the kernel API.

    /// IDCT constants — additional pair-sums beyond the fdct table.
    /// Layout mirrors libjpeg-turbo `jconst_idct_islow_avx2` SEG_CONST.
    static IDCT_PW_MF089_F060_MF256_F050: Aligned32<[i16; 16]> = Aligned32([
        -7373, 4926, -7373, 4926, -7373, 4926, -7373, 4926, -20995, 4177, -20995, 4177, -20995,
        4177, -20995, 4177,
    ]);

    /// Descale round-bias for the IDCT's final pass-2 shift
    /// (1 << (18-1) = 131072).
    static IDCT_PD_DESCALE_P2: Aligned32<[i32; 8]> = Aligned32([131072; 8]);
    // IDCT_PD_DESCALE_P1 is identical to fdct's PD_DESCALE_P1 (1024).
    // Reuse it directly.

    /// 1D IDCT pass. Const-generic on PASS (1 or 2) selects descale
    /// shift (11 for pass 1 → next pass's i16 workspace; 18 for pass 2
    /// → final pixel-domain output). Returns 4 ymm of i16, each holding
    /// 2 packed rows (low/high 128).
    ///
    /// In/out register naming follows upstream:
    ///   in0_4 = (col0, col4), in3_1 = (col3, col1)
    ///   in2_6 = (col2, col6), in7_5 = (col7, col5)
    /// returns (data0_1, data3_2, data4_5, data7_6)
    #[inline(always)]
    unsafe fn dodct_idct<const PASS: i32>(
        in0_4: __m256i,
        in3_1: __m256i,
        in2_6: __m256i,
        in7_5: __m256i,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        unsafe {
            // ---- Even part ----
            let in6_2 = _mm256_permute2x128_si256::<0x01>(in2_6, in2_6);
            let in26l = _mm256_unpacklo_epi16(in2_6, in6_2);
            let in26h = _mm256_unpackhi_epi16(in2_6, in6_2);
            let pw_f130 = load(PW_F130_F054_MF130_F054.0.as_ptr());
            let tmp32l = _mm256_madd_epi16(in26l, pw_f130); // tmp3_2 L
            let tmp32h = _mm256_madd_epi16(in26h, pw_f130); // tmp3_2 H

            // (in0+in4) and (in0-in4), via sign-flip + permute trick.
            let in4_0 = _mm256_permute2x128_si256::<0x01>(in0_4, in0_4);
            let pw_1_neg1 = load(PW_1_NEG1.0.as_ptr());
            // Negate the high half: yields (in0, -in4).
            let in0_neg4 = _mm256_sign_epi16(in0_4, pw_1_neg1);
            // Add → (in0+in4, in0-in4).
            let tmp01_s16 = _mm256_add_epi16(in4_0, in0_neg4);

            // Widen i16 → i32 with left-shift-by-CONST_BITS via unpack +
            // arithmetic right shift (sign-extension trick — see
            // upstream lines 142-145).
            let zero = _mm256_setzero_si256();
            let tmp01l = _mm256_unpacklo_epi16(zero, tmp01_s16);
            let tmp01h = _mm256_unpackhi_epi16(zero, tmp01_s16);
            // Arithmetic right shift by (16 - CONST_BITS) = 3
            // (equivalent to having sign-extended via lo unpack
            // then left-shifted by CONST_BITS).
            let tmp01l = _mm256_srai_epi32::<3>(tmp01l);
            let tmp01h = _mm256_srai_epi32::<3>(tmp01h);

            let tmp10_11l = _mm256_add_epi32(tmp01l, tmp32l);
            let tmp10_11h = _mm256_add_epi32(tmp01h, tmp32h);
            let tmp13_12l = _mm256_sub_epi32(tmp01l, tmp32l);
            let tmp13_12h = _mm256_sub_epi32(tmp01h, tmp32h);

            // ---- Odd part ----
            // z3_4 = (in7+in3, in5+in1)
            let z3_4 = _mm256_add_epi16(in7_5, in3_1);

            let z4_3 = _mm256_permute2x128_si256::<0x01>(z3_4, z3_4);
            let z34l = _mm256_unpacklo_epi16(z3_4, z4_3);
            let z34h = _mm256_unpackhi_epi16(z3_4, z4_3);
            let pw_mf078 = load(PW_MF078_F117_F078_F117.0.as_ptr());
            // z3_4 = z3 * (F_1_175 - F_1_961) + z4 * F_1_175
            //      = z4 * F_1_175 + z3 * (F_1_175 - F_0_390) etc
            let z34_l = _mm256_madd_epi16(z34l, pw_mf078);
            let z34_h = _mm256_madd_epi16(z34h, pw_mf078);

            let in1_3 = _mm256_permute2x128_si256::<0x01>(in3_1, in3_1);
            let in71_53l = _mm256_unpacklo_epi16(in7_5, in1_3);
            let in71_53h = _mm256_unpackhi_epi16(in7_5, in1_3);

            // tmp0_1 = in7 * (F_0_298 - F_0_899) + in1 * (-F_0_899)
            //        | in5 * (F_2_053 - F_2_562) + in3 * (-F_2_562)
            let pw_mf060 = load(PW_MF060_MF089_MF050_MF256.0.as_ptr());
            let tmp01l_o = _mm256_madd_epi16(in71_53l, pw_mf060);
            let tmp01h_o = _mm256_madd_epi16(in71_53h, pw_mf060);
            let tmp01l_o = _mm256_add_epi32(tmp01l_o, z34_l);
            let tmp01h_o = _mm256_add_epi32(tmp01h_o, z34_h);

            // tmp3_2 = in7 * (-F_0_899) + in1 * (F_1_501 - F_0_899)
            //        | in5 * (-F_2_562) + in3 * (F_3_072 - F_2_562)
            let pw_mf089 = load(IDCT_PW_MF089_F060_MF256_F050.0.as_ptr());
            let tmp32l_o = _mm256_madd_epi16(in71_53l, pw_mf089);
            let tmp32h_o = _mm256_madd_epi16(in71_53h, pw_mf089);
            // Swap halves of z3_4 to get z4_3.
            let z4_3l = _mm256_permute2x128_si256::<0x01>(z34_l, z34_l);
            let z4_3h = _mm256_permute2x128_si256::<0x01>(z34_h, z34_h);
            let tmp32l_o = _mm256_add_epi32(tmp32l_o, z4_3l);
            let tmp32h_o = _mm256_add_epi32(tmp32h_o, z4_3h);

            // ---- Final output stage ----
            // data0_1 = tmp10_11 + tmp3_2 (descaled)
            // data7_6 = tmp10_11 - tmp3_2 (descaled)
            // data3_2 = tmp13_12 + tmp0_1 (descaled)
            // data4_5 = tmp13_12 - tmp0_1 (descaled)
            let pd_descale = if PASS == 1 {
                _mm256_loadu_si256(PD_DESCALE_P1.0.as_ptr() as *const __m256i)
            } else {
                _mm256_loadu_si256(IDCT_PD_DESCALE_P2.0.as_ptr() as *const __m256i)
            };

            let d01l = _mm256_add_epi32(_mm256_add_epi32(tmp10_11l, tmp32l_o), pd_descale);
            let d01h = _mm256_add_epi32(_mm256_add_epi32(tmp10_11h, tmp32h_o), pd_descale);
            let d76l = _mm256_add_epi32(_mm256_sub_epi32(tmp10_11l, tmp32l_o), pd_descale);
            let d76h = _mm256_add_epi32(_mm256_sub_epi32(tmp10_11h, tmp32h_o), pd_descale);
            let d32l = _mm256_add_epi32(_mm256_add_epi32(tmp13_12l, tmp01l_o), pd_descale);
            let d32h = _mm256_add_epi32(_mm256_add_epi32(tmp13_12h, tmp01h_o), pd_descale);
            let d45l = _mm256_add_epi32(_mm256_sub_epi32(tmp13_12l, tmp01l_o), pd_descale);
            let d45h = _mm256_add_epi32(_mm256_sub_epi32(tmp13_12h, tmp01h_o), pd_descale);

            let (d01l, d01h, d76l, d76h, d32l, d32h, d45l, d45h) = if PASS == 1 {
                (
                    _mm256_srai_epi32::<11>(d01l), _mm256_srai_epi32::<11>(d01h),
                    _mm256_srai_epi32::<11>(d76l), _mm256_srai_epi32::<11>(d76h),
                    _mm256_srai_epi32::<11>(d32l), _mm256_srai_epi32::<11>(d32h),
                    _mm256_srai_epi32::<11>(d45l), _mm256_srai_epi32::<11>(d45h),
                )
            } else {
                (
                    _mm256_srai_epi32::<18>(d01l), _mm256_srai_epi32::<18>(d01h),
                    _mm256_srai_epi32::<18>(d76l), _mm256_srai_epi32::<18>(d76h),
                    _mm256_srai_epi32::<18>(d32l), _mm256_srai_epi32::<18>(d32h),
                    _mm256_srai_epi32::<18>(d45l), _mm256_srai_epi32::<18>(d45h),
                )
            };

            // Pack i32 → i16 (saturating). Pairs L+H per output.
            let data0_1 = _mm256_packs_epi32(d01l, d01h);
            let data7_6 = _mm256_packs_epi32(d76l, d76h);
            let data3_2 = _mm256_packs_epi32(d32l, d32h);
            let data4_5 = _mm256_packs_epi32(d45l, d45h);

            (data0_1, data3_2, data4_5, data7_6)
        }
    }

    /// AVX2 inverse DCT. Reads quantized i16 + qt, writes i32 workspace
    /// (pre `+128` shift, pre clamp).
    ///
    /// # Safety
    /// Caller must have AVX2 available (runtime check in
    /// `idct_kernel`'s dispatcher).
    #[target_feature(enable = "avx2")]
    pub(in crate::codec::jpeg) unsafe fn idct_kernel_avx2(
        quantized: &[i16; 64],
        qt: &[u16; 64],
        out: &mut [i32; 64],
    ) {
        let qp = quantized.as_ptr() as *const __m256i;
        let tp = qt.as_ptr() as *const __m256i; // u16 → i16 (values ≤ 255)

        // Load 4 ymm of i16 coefficients (rows 0_1, 2_3, 4_5, 6_7).
        let in0_1 = _mm256_loadu_si256(qp);
        let in2_3 = _mm256_loadu_si256(qp.add(1));
        let in4_5 = _mm256_loadu_si256(qp.add(2));
        let in6_7 = _mm256_loadu_si256(qp.add(3));

        // Dequantize (i16 * i16 → i16 wrapping; values fit in i16 for
        // valid 8-bit JPEG — matches upstream's vpmullw).
        let in0_1 = _mm256_mullo_epi16(in0_1, _mm256_loadu_si256(tp));
        let in2_3 = _mm256_mullo_epi16(in2_3, _mm256_loadu_si256(tp.add(1)));
        let in4_5 = _mm256_mullo_epi16(in4_5, _mm256_loadu_si256(tp.add(2)));
        let in6_7 = _mm256_mullo_epi16(in6_7, _mm256_loadu_si256(tp.add(3)));

        // Re-pack to (in0_4, in3_1, in2_6, in7_5) — upstream pattern.
        let in0_4 = _mm256_permute2x128_si256::<0x20>(in0_1, in4_5);
        let in3_1 = _mm256_permute2x128_si256::<0x31>(in2_3, in0_1);
        let in2_6 = _mm256_permute2x128_si256::<0x20>(in2_3, in6_7);
        let in7_5 = _mm256_permute2x128_si256::<0x31>(in6_7, in4_5);

        // Pass 1.
        let (d0_1, d3_2, d4_5, d7_6) = dodct_idct::<1>(in0_4, in3_1, in2_6, in7_5);

        // Transpose: 4 ymm (2 rows packed) → 4 ymm (2 cols packed).
        let (t0_4, t1_5, t2_6, t3_7) = dotranspose(d0_1, d3_2, d4_5, d7_6);

        // Re-pack for pass 2: pass 2 wants (in0_4, in3_1, in2_6, in7_5)
        // again. After dotranspose: (col0_4, col1_5, col2_6, col3_7).
        // Need to compute (in7_5, in3_1) from those.
        let in7_5_p2 = _mm256_permute2x128_si256::<0x31>(t3_7, t1_5);
        let in3_1_p2 = _mm256_permute2x128_si256::<0x20>(t3_7, t1_5);

        // Pass 2.
        let (d0_1, d3_2, d4_5, d7_6) = dodct_idct::<2>(t0_4, in3_1_p2, t2_6, in7_5_p2);

        // Transpose back: 4 ymm packed (row N, row N+4) → row-major.
        let (r0_4, r1_5, r2_6, r3_7) = dotranspose(d0_1, d3_2, d4_5, d7_6);

        // Each ymm holds 2 rows packed (low-128 = row N i16x8, high-128
        // = row N+4 i16x8). Unpack i16 → i32 with sign-extend, store as
        // 8 i32 per row.
        let out_ptr = out.as_mut_ptr() as *mut __m256i;
        store_row_as_i32(r0_4, out_ptr.add(0), out_ptr.add(8));
        store_row_as_i32(r1_5, out_ptr.add(2), out_ptr.add(10));
        store_row_as_i32(r2_6, out_ptr.add(4), out_ptr.add(12));
        store_row_as_i32(r3_7, out_ptr.add(6), out_ptr.add(14));

        _mm256_zeroupper();
    }

    /// Unpack the low-128 and high-128 halves of a packed (rowN, rowN+4)
    /// ymm to i32x8 vectors, store to two i32x8-sized destinations.
    #[inline(always)]
    unsafe fn store_row_as_i32(packed: __m256i, dst_lo: *mut __m256i, dst_hi: *mut __m256i) {
        unsafe {
            // Extract the two 128-bit halves as i16x8.
            let lo_i16 = _mm256_castsi256_si128(packed);
            let hi_i16 = _mm256_extracti128_si256::<1>(packed);
            // Sign-extend i16 → i32 to fill the ymm.
            let lo_i32 = _mm256_cvtepi16_epi32(lo_i16);
            let hi_i32 = _mm256_cvtepi16_epi32(hi_i16);
            _mm256_storeu_si256(dst_lo, lo_i32);
            _mm256_storeu_si256(dst_hi, hi_i32);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) mod wasm {
    //! WASM SIMD128 LL&M DCT/IDCT (wasm32).
    //!
    //! No Rust reference exists for LL&M in WASM SIMD128 (jpeg-rusturbo
    //! has no WASM target; image-rs/jpeg-decoder's WASM IDCT uses
    //! stb-style constants, not LL&M). Written from libjpeg-turbo
    //! upstream + my own NEON/AVX2 ports as references.
    //!
    //! Structurally identical to my NEON kernel (`mod neon` above) —
    //! both are 128-bit SIMD with 8 i16 lanes per vector. The
    //! intrinsic mapping is 1:1 except:
    //!
    //! - NEON's `vmull_lane_s16<L>(a, broadcast_reg)` (broadcast-multiply
    //!   widening) → WASM has no lane-broadcast. Use
    //!   `i16x8_splat(C)` once per constant + `i32x4_extmul_low_i16x8`
    //!   / `_high` for the two halves.
    //! - NEON's `vrshrn_n_s32<N>(a)` (rounding narrow) → WASM
    //!   `i16x8_narrow_i32x4` doesn't include rounding; pre-add the
    //!   round bias then shift then narrow.
    //!
    //! Constants are the same Q13 LL&M values, just held as splatted
    //! v128 instead of packed lanes.

    use core::arch::wasm32::*;

    // ============================================================
    // Q13 LL&M constants (same as NEON / AVX2 / scalar).
    // ============================================================

    const F_0_298: i16 = 2446;
    const F_0_390: i16 = 3196;
    const F_0_541: i16 = 4433;
    const F_0_765: i16 = 6270;
    const F_0_899: i16 = 7373;
    const F_1_175: i16 = 9633;
    const F_1_501: i16 = 12299;
    const F_1_847: i16 = 15137;
    const F_1_961: i16 = 16069;
    const F_2_053: i16 = 16819;
    const F_2_562: i16 = 20995;
    const F_3_072: i16 = 25172;

    /// Round-shift right by N + narrow i32→i16 with saturation.
    /// (`vrshrn_n_s32<N>` in NEON; not a single instruction in WASM.)
    #[inline(always)]
    fn descale_narrow<const N: u32>(lo: v128, hi: v128) -> v128 {
        let bias = i32x4_splat(1 << (N - 1));
        let lo = i32x4_shr(i32x4_add(lo, bias), N);
        let hi = i32x4_shr(i32x4_add(hi, bias), N);
        i16x8_narrow_i32x4(lo, hi)
    }

    /// Multiply each i16 lane of `a` by the i16 scalar constant `c`,
    /// widening to two i32x4 (low half, high half). Acts like NEON's
    /// `vmull_n_s16(a, c)` — but WASM SIMD doesn't have an n-form, so
    /// we splat the constant and use `i32x4_extmul`.
    ///
    /// Returns (low4, high4) i32x4 widened products.
    #[inline(always)]
    fn vmull_n(a: v128, splat_c: v128) -> (v128, v128) {
        (
            i32x4_extmul_low_i16x8(a, splat_c),
            i32x4_extmul_high_i16x8(a, splat_c),
        )
    }

    /// As `vmull_n` but adds to an existing accumulator (NEON's vmlal_n).
    #[inline(always)]
    fn vmlal_n(acc_lo: v128, acc_hi: v128, a: v128, splat_c: v128) -> (v128, v128) {
        let (lo, hi) = vmull_n(a, splat_c);
        (i32x4_add(acc_lo, lo), i32x4_add(acc_hi, hi))
    }

    /// 8×8 transpose of i16 lanes across 8 v128 registers. Each input
    /// `r0..r7` holds 8 i16 in lane order; on output each holds the
    /// corresponding column. Uses 3 stages of `i16x8_shuffle` (≈ the
    /// standard NEON `vtrn/vzip` pattern hand-emulated).
    ///
    /// Lane index convention: `i16x8_shuffle::<I0..I7>(a, b)` — output
    /// lane `j` = a-or-b lane `Ij` (0..7 = a, 8..15 = b).
    #[inline(always)]
    fn transpose_8x8_i16(rows: &mut [v128; 8]) {
        let &mut [r0, r1, r2, r3, r4, r5, r6, r7] = rows;

        // Stage 1: interleave adjacent 16-bit lanes pairwise.
        // After: each register holds (rA[0], rB[0], rA[1], rB[1], ...).
        let t0 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(r0, r1); // 16-bit pairs
        let t1 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(r0, r1);
        let t2 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(r2, r3);
        let t3 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(r2, r3);
        let t4 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(r4, r5);
        let t5 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(r4, r5);
        let t6 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(r6, r7);
        let t7 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(r6, r7);

        // Stage 2: interleave 32-bit lanes (pairs of 16-bit lanes).
        // After: each register holds 4 consecutive cols × 2 rows.
        // Reinterpret as i32 view via i32x4_shuffle.
        let u0 = i32x4_shuffle::<0, 4, 1, 5>(t0, t2); // pairs of (r0r1, r2r3) at cols 0..3
        let u1 = i32x4_shuffle::<2, 6, 3, 7>(t0, t2);
        let u2 = i32x4_shuffle::<0, 4, 1, 5>(t1, t3); // cols 4..7
        let u3 = i32x4_shuffle::<2, 6, 3, 7>(t1, t3);
        let u4 = i32x4_shuffle::<0, 4, 1, 5>(t4, t6); // pairs of (r4r5, r6r7) at cols 0..3
        let u5 = i32x4_shuffle::<2, 6, 3, 7>(t4, t6);
        let u6 = i32x4_shuffle::<0, 4, 1, 5>(t5, t7); // cols 4..7
        let u7 = i32x4_shuffle::<2, 6, 3, 7>(t5, t7);

        // Stage 3: interleave 64-bit halves.
        // After: each register holds one full column (rows 0..7).
        rows[0] = i64x2_shuffle::<0, 2>(u0, u4); // col 0
        rows[1] = i64x2_shuffle::<1, 3>(u0, u4); // col 1
        rows[2] = i64x2_shuffle::<0, 2>(u1, u5); // col 2
        rows[3] = i64x2_shuffle::<1, 3>(u1, u5); // col 3
        rows[4] = i64x2_shuffle::<0, 2>(u2, u6); // col 4
        rows[5] = i64x2_shuffle::<1, 3>(u2, u6); // col 5
        rows[6] = i64x2_shuffle::<0, 2>(u3, u7); // col 6
        rows[7] = i64x2_shuffle::<1, 3>(u3, u7); // col 7
    }

    /// WASM SIMD128 forward DCT, in-place. Bit-exact equivalent to
    /// `fdct_kernel_scalar`. Layout mirrors `neon::fdct_kernel_neon`
    /// — 8 v128 each holding 1 row.
    ///
    /// # Safety
    /// `target_feature = "simd128"` enabled (via `.cargo/config.toml`'s
    /// rustflags for wasm32 + the `#[target_feature]` attribute).
    #[target_feature(enable = "simd128")]
    pub(in crate::codec::jpeg) unsafe fn fdct_kernel_wasm(data: &mut [i16; 64]) {
        // Splat all Q13 constants we'll need.
        let c_298 = i16x8_splat(F_0_298);
        let c_390n = i16x8_splat(-F_0_390);
        let c_541 = i16x8_splat(F_0_541);
        let c_765 = i16x8_splat(F_0_765);
        let c_899n = i16x8_splat(-F_0_899);
        let c_1175 = i16x8_splat(F_1_175);
        let c_1501 = i16x8_splat(F_1_501);
        let c_1847n = i16x8_splat(-F_1_847);
        let c_1961n = i16x8_splat(-F_1_961);
        let c_2053 = i16x8_splat(F_2_053);
        let c_2562n = i16x8_splat(-F_2_562);
        let c_3072 = i16x8_splat(F_3_072);

        // Pass 1 — load 8 rows of input.
        let p = data.as_mut_ptr() as *const v128;
        let mut rows: [v128; 8] = [
            v128_load(p),
            v128_load(p.add(1)),
            v128_load(p.add(2)),
            v128_load(p.add(3)),
            v128_load(p.add(4)),
            v128_load(p.add(5)),
            v128_load(p.add(6)),
            v128_load(p.add(7)),
        ];

        // Transpose so each v128 holds one column (across all 8 rows).
        transpose_8x8_i16(&mut rows);
        let [c0, c1, c2, c3, c4, c5, c6, c7] = rows;

        // Pass 1 1D butterfly across columns (= across rows pre-transpose).
        let out_p1 = fdct_1d_pass1(
            c0, c1, c2, c3, c4, c5, c6, c7,
            c_541, c_765, c_1847n, c_298, c_1175, c_1501, c_2053, c_3072,
            c_899n, c_2562n, c_1961n, c_390n,
        );
        let mut rows = out_p1;

        // Transpose back so each v128 holds one row.
        transpose_8x8_i16(&mut rows);
        let [c0, c1, c2, c3, c4, c5, c6, c7] = rows;

        // Pass 2 1D butterfly.
        let out_p2 = fdct_1d_pass2(
            c0, c1, c2, c3, c4, c5, c6, c7,
            c_541, c_765, c_1847n, c_298, c_1175, c_1501, c_2053, c_3072,
            c_899n, c_2562n, c_1961n, c_390n,
        );

        // Store.
        let q = data.as_mut_ptr() as *mut v128;
        v128_store(q, out_p2[0]);
        v128_store(q.add(1), out_p2[1]);
        v128_store(q.add(2), out_p2[2]);
        v128_store(q.add(3), out_p2[3]);
        v128_store(q.add(4), out_p2[4]);
        v128_store(q.add(5), out_p2[5]);
        v128_store(q.add(6), out_p2[6]);
        v128_store(q.add(7), out_p2[7]);
    }

    /// One 1D fdct pass (pass 1 — outputs scaled by 2^PASS1_BITS).
    /// `c_*` are splatted Q13 constants.
    #[inline(always)]
    fn fdct_1d_pass1(
        c0: v128, c1: v128, c2: v128, c3: v128, c4: v128, c5: v128, c6: v128, c7: v128,
        c_541: v128, c_765: v128, c_1847n: v128, c_298: v128, c_1175: v128,
        c_1501: v128, c_2053: v128, c_3072: v128,
        c_899n: v128, c_2562n: v128, c_1961n: v128, c_390n: v128,
    ) -> [v128; 8] {
        const PASS1_BITS: u32 = 2;
        const DESCALE_P1: u32 = 11; // CONST_BITS - PASS1_BITS

        let tmp0 = i16x8_add(c0, c7);
        let tmp7 = i16x8_sub(c0, c7);
        let tmp1 = i16x8_add(c1, c6);
        let tmp6 = i16x8_sub(c1, c6);
        let tmp2 = i16x8_add(c2, c5);
        let tmp5 = i16x8_sub(c2, c5);
        let tmp3 = i16x8_add(c3, c4);
        let tmp4 = i16x8_sub(c3, c4);

        // Even part
        let tmp10 = i16x8_add(tmp0, tmp3);
        let tmp13 = i16x8_sub(tmp0, tmp3);
        let tmp11 = i16x8_add(tmp1, tmp2);
        let tmp12 = i16x8_sub(tmp1, tmp2);

        let out_0 = i16x8_shl(i16x8_add(tmp10, tmp11), PASS1_BITS);
        let out_4 = i16x8_shl(i16x8_sub(tmp10, tmp11), PASS1_BITS);

        let tmp1213 = i16x8_add(tmp12, tmp13);
        let (z1_l, z1_h) = vmull_n(tmp1213, c_541);
        let (col2_l, col2_h) = vmlal_n(z1_l, z1_h, tmp13, c_765);
        let out_2 = descale_narrow::<DESCALE_P1>(col2_l, col2_h);
        let (col6_l, col6_h) = vmlal_n(z1_l, z1_h, tmp12, c_1847n);
        let out_6 = descale_narrow::<DESCALE_P1>(col6_l, col6_h);

        // Odd part
        let z1o = i16x8_add(tmp4, tmp7);
        let z2o = i16x8_add(tmp5, tmp6);
        let z3o = i16x8_add(tmp4, tmp6);
        let z4o = i16x8_add(tmp5, tmp7);
        let z34 = i16x8_add(z3o, z4o);
        let (z5_l, z5_h) = vmull_n(z34, c_1175);

        let (t4_l, t4_h) = vmull_n(tmp4, c_298);
        let (t5_l, t5_h) = vmull_n(tmp5, c_2053);
        let (t6_l, t6_h) = vmull_n(tmp6, c_3072);
        let (t7_l, t7_h) = vmull_n(tmp7, c_1501);

        let (z1p_l, z1p_h) = vmull_n(z1o, c_899n);
        let (z2p_l, z2p_h) = vmull_n(z2o, c_2562n);
        let (z3p_l_pre, z3p_h_pre) = vmull_n(z3o, c_1961n);
        let z3p_l = i32x4_add(z3p_l_pre, z5_l);
        let z3p_h = i32x4_add(z3p_h_pre, z5_h);
        let (z4p_l_pre, z4p_h_pre) = vmull_n(z4o, c_390n);
        let z4p_l = i32x4_add(z4p_l_pre, z5_l);
        let z4p_h = i32x4_add(z4p_h_pre, z5_h);

        let out_7_l = i32x4_add(i32x4_add(t4_l, z1p_l), z3p_l);
        let out_7_h = i32x4_add(i32x4_add(t4_h, z1p_h), z3p_h);
        let out_7 = descale_narrow::<DESCALE_P1>(out_7_l, out_7_h);

        let out_5_l = i32x4_add(i32x4_add(t5_l, z2p_l), z4p_l);
        let out_5_h = i32x4_add(i32x4_add(t5_h, z2p_h), z4p_h);
        let out_5 = descale_narrow::<DESCALE_P1>(out_5_l, out_5_h);

        let out_3_l = i32x4_add(i32x4_add(t6_l, z2p_l), z3p_l);
        let out_3_h = i32x4_add(i32x4_add(t6_h, z2p_h), z3p_h);
        let out_3 = descale_narrow::<DESCALE_P1>(out_3_l, out_3_h);

        let out_1_l = i32x4_add(i32x4_add(t7_l, z1p_l), z4p_l);
        let out_1_h = i32x4_add(i32x4_add(t7_h, z1p_h), z4p_h);
        let out_1 = descale_narrow::<DESCALE_P1>(out_1_l, out_1_h);

        [out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7]
    }

    /// One 1D fdct pass (pass 2 — descale by CONST_BITS + PASS1_BITS).
    /// Identical to pass 1 except DC outputs descale by PASS1_BITS
    /// (with rounding) and multiplied outputs descale by 15.
    #[inline(always)]
    fn fdct_1d_pass2(
        c0: v128, c1: v128, c2: v128, c3: v128, c4: v128, c5: v128, c6: v128, c7: v128,
        c_541: v128, c_765: v128, c_1847n: v128, c_298: v128, c_1175: v128,
        c_1501: v128, c_2053: v128, c_3072: v128,
        c_899n: v128, c_2562n: v128, c_1961n: v128, c_390n: v128,
    ) -> [v128; 8] {
        const PASS1_BITS: u32 = 2;
        const DESCALE_P2: u32 = 15; // CONST_BITS + PASS1_BITS

        let tmp0 = i16x8_add(c0, c7);
        let tmp7 = i16x8_sub(c0, c7);
        let tmp1 = i16x8_add(c1, c6);
        let tmp6 = i16x8_sub(c1, c6);
        let tmp2 = i16x8_add(c2, c5);
        let tmp5 = i16x8_sub(c2, c5);
        let tmp3 = i16x8_add(c3, c4);
        let tmp4 = i16x8_sub(c3, c4);

        // Even part
        let tmp10 = i16x8_add(tmp0, tmp3);
        let tmp13 = i16x8_sub(tmp0, tmp3);
        let tmp11 = i16x8_add(tmp1, tmp2);
        let tmp12 = i16x8_sub(tmp1, tmp2);

        // Pass-2 DC outputs: descale by PASS1_BITS with rounding.
        let bias = i16x8_splat(1 << (PASS1_BITS - 1));
        let out_0 = i16x8_shr(i16x8_add(i16x8_add(tmp10, tmp11), bias), PASS1_BITS);
        let out_4 = i16x8_shr(i16x8_add(i16x8_sub(tmp10, tmp11), bias), PASS1_BITS);

        let tmp1213 = i16x8_add(tmp12, tmp13);
        let (z1_l, z1_h) = vmull_n(tmp1213, c_541);
        let (col2_l, col2_h) = vmlal_n(z1_l, z1_h, tmp13, c_765);
        let out_2 = descale_narrow::<DESCALE_P2>(col2_l, col2_h);
        let (col6_l, col6_h) = vmlal_n(z1_l, z1_h, tmp12, c_1847n);
        let out_6 = descale_narrow::<DESCALE_P2>(col6_l, col6_h);

        // Odd part
        let z1o = i16x8_add(tmp4, tmp7);
        let z2o = i16x8_add(tmp5, tmp6);
        let z3o = i16x8_add(tmp4, tmp6);
        let z4o = i16x8_add(tmp5, tmp7);
        let z34 = i16x8_add(z3o, z4o);
        let (z5_l, z5_h) = vmull_n(z34, c_1175);

        let (t4_l, t4_h) = vmull_n(tmp4, c_298);
        let (t5_l, t5_h) = vmull_n(tmp5, c_2053);
        let (t6_l, t6_h) = vmull_n(tmp6, c_3072);
        let (t7_l, t7_h) = vmull_n(tmp7, c_1501);

        let (z1p_l, z1p_h) = vmull_n(z1o, c_899n);
        let (z2p_l, z2p_h) = vmull_n(z2o, c_2562n);
        let (z3p_l_pre, z3p_h_pre) = vmull_n(z3o, c_1961n);
        let z3p_l = i32x4_add(z3p_l_pre, z5_l);
        let z3p_h = i32x4_add(z3p_h_pre, z5_h);
        let (z4p_l_pre, z4p_h_pre) = vmull_n(z4o, c_390n);
        let z4p_l = i32x4_add(z4p_l_pre, z5_l);
        let z4p_h = i32x4_add(z4p_h_pre, z5_h);

        let out_7_l = i32x4_add(i32x4_add(t4_l, z1p_l), z3p_l);
        let out_7_h = i32x4_add(i32x4_add(t4_h, z1p_h), z3p_h);
        let out_7 = descale_narrow::<DESCALE_P2>(out_7_l, out_7_h);

        let out_5_l = i32x4_add(i32x4_add(t5_l, z2p_l), z4p_l);
        let out_5_h = i32x4_add(i32x4_add(t5_h, z2p_h), z4p_h);
        let out_5 = descale_narrow::<DESCALE_P2>(out_5_l, out_5_h);

        let out_3_l = i32x4_add(i32x4_add(t6_l, z2p_l), z3p_l);
        let out_3_h = i32x4_add(i32x4_add(t6_h, z2p_h), z3p_h);
        let out_3 = descale_narrow::<DESCALE_P2>(out_3_l, out_3_h);

        let out_1_l = i32x4_add(i32x4_add(t7_l, z1p_l), z4p_l);
        let out_1_h = i32x4_add(i32x4_add(t7_h, z1p_h), z4p_h);
        let out_1 = descale_narrow::<DESCALE_P2>(out_1_l, out_1_h);

        [out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7]
    }

    // ====================================================================
    // IDCT — WASM SIMD128 inverse DCT
    // ====================================================================

    /// WASM SIMD128 inverse DCT. Reads quantized i16 + qt, writes i32
    /// workspace (pre `+128` shift, pre clamp).
    ///
    /// Structure: 8-wide (one v128 per row/column). Dequant + transpose
    /// + pass-1 IDCT narrowing to i16 workspace + transpose + pass-2
    /// IDCT narrowing to i16, then unpack each row's i16x8 to two
    /// i32x4 v128 for the i32 output store.
    ///
    /// # Safety
    /// simd128 enabled.
    #[target_feature(enable = "simd128")]
    pub(in crate::codec::jpeg) unsafe fn idct_kernel_wasm(
        quantized: &[i16; 64],
        qt: &[u16; 64],
        out: &mut [i32; 64],
    ) {
        let c_298 = i16x8_splat(F_0_298);
        let c_390n = i16x8_splat(-F_0_390);
        let c_541 = i16x8_splat(F_0_541);
        let c_765 = i16x8_splat(F_0_765);
        let c_899n = i16x8_splat(-F_0_899);
        let c_1175 = i16x8_splat(F_1_175);
        let c_1501 = i16x8_splat(F_1_501);
        let c_1847n = i16x8_splat(-F_1_847);
        let c_1961n = i16x8_splat(-F_1_961);
        let c_2053 = i16x8_splat(F_2_053);
        let c_2562n = i16x8_splat(-F_2_562);
        let c_3072 = i16x8_splat(F_3_072);

        // Load 8 rows of coefficients + 8 rows of qt.
        let qp = quantized.as_ptr() as *const v128;
        let tp = qt.as_ptr() as *const v128;
        // Dequant inline: i16 lane-wise multiply (wrapping; same as
        // NEON `vmul_s16` and AVX2 `vpmullw` — values fit in i16 for
        // valid 8-bit JPEG).
        let mut rows: [v128; 8] = [
            i16x8_mul(v128_load(qp), v128_load(tp)),
            i16x8_mul(v128_load(qp.add(1)), v128_load(tp.add(1))),
            i16x8_mul(v128_load(qp.add(2)), v128_load(tp.add(2))),
            i16x8_mul(v128_load(qp.add(3)), v128_load(tp.add(3))),
            i16x8_mul(v128_load(qp.add(4)), v128_load(tp.add(4))),
            i16x8_mul(v128_load(qp.add(5)), v128_load(tp.add(5))),
            i16x8_mul(v128_load(qp.add(6)), v128_load(tp.add(6))),
            i16x8_mul(v128_load(qp.add(7)), v128_load(tp.add(7))),
        ];

        // IDCT pass 1 is COLUMN-wise (LL&M / libjpeg-turbo convention,
        // opposite of fdct's row-wise pass 1). With register-per-row
        // layout (lane k = col k), cross-vector butterfly combines
        // values at the same column across rows — that's exactly the
        // column-wise IDCT we want. **No initial transpose.**
        let [c0, c1, c2, c3, c4, c5, c6, c7] = rows;

        // Pass 1: column-wise IDCT, descale by 11, narrow to i16.
        let out_p1 = idct_1d_pass1(
            c0, c1, c2, c3, c4, c5, c6, c7,
            c_541, c_765, c_1847n, c_298, c_1175, c_1501, c_2053, c_3072,
            c_899n, c_2562n, c_1961n, c_390n,
        );
        let mut rows = out_p1;

        // Transpose so each v128 holds one column of post-pass-1, i.e.
        // (lane k = row k). Pass 2 cross-vector butterfly then combines
        // values at the same row across columns — row-wise IDCT.
        transpose_8x8_i16(&mut rows);
        let [c0, c1, c2, c3, c4, c5, c6, c7] = rows;

        // Pass 2: row-wise IDCT, descale by 18, narrow to i16.
        let out_p2 = idct_1d_pass2(
            c0, c1, c2, c3, c4, c5, c6, c7,
            c_541, c_765, c_1847n, c_298, c_1175, c_1501, c_2053, c_3072,
            c_899n, c_2562n, c_1961n, c_390n,
        );

        // Transpose back: post-pass-2 registers hold "col of final
        // output, lane = row". Transpose so each v128 holds one row.
        let mut rows = out_p2;
        transpose_8x8_i16(&mut rows);

        // Unpack each row's i16x8 to two i32x4 (low + high halves),
        // store as i32 workspace.
        let out_ptr = out.as_mut_ptr() as *mut v128;
        for (i, row) in rows.iter().enumerate() {
            let lo = i32x4_extend_low_i16x8(*row);
            let hi = i32x4_extend_high_i16x8(*row);
            v128_store(out_ptr.add(2 * i), lo);
            v128_store(out_ptr.add(2 * i + 1), hi);
        }
    }

    /// 1D IDCT pass 1 (descale by CONST_BITS - PASS1_BITS = 11, narrow
    /// to i16 workspace).
    #[inline(always)]
    fn idct_1d_pass1(
        c0: v128, c1: v128, c2: v128, c3: v128, c4: v128, c5: v128, c6: v128, c7: v128,
        c_541: v128, c_765: v128, c_1847n: v128, c_298: v128, c_1175: v128,
        c_1501: v128, c_2053: v128, c_3072: v128,
        c_899n: v128, c_2562n: v128, c_1961n: v128, c_390n: v128,
    ) -> [v128; 8] {
        const DESCALE_P1: u32 = 11;
        idct_1d_inner::<DESCALE_P1>(
            c0, c1, c2, c3, c4, c5, c6, c7,
            c_541, c_765, c_1847n, c_298, c_1175, c_1501, c_2053, c_3072,
            c_899n, c_2562n, c_1961n, c_390n,
        )
    }

    /// 1D IDCT pass 2 (descale by CONST_BITS + PASS1_BITS + 3 = 18,
    /// narrow to i16 pixel-domain output).
    #[inline(always)]
    fn idct_1d_pass2(
        c0: v128, c1: v128, c2: v128, c3: v128, c4: v128, c5: v128, c6: v128, c7: v128,
        c_541: v128, c_765: v128, c_1847n: v128, c_298: v128, c_1175: v128,
        c_1501: v128, c_2053: v128, c_3072: v128,
        c_899n: v128, c_2562n: v128, c_1961n: v128, c_390n: v128,
    ) -> [v128; 8] {
        const DESCALE_P2: u32 = 18;
        idct_1d_inner::<DESCALE_P2>(
            c0, c1, c2, c3, c4, c5, c6, c7,
            c_541, c_765, c_1847n, c_298, c_1175, c_1501, c_2053, c_3072,
            c_899n, c_2562n, c_1961n, c_390n,
        )
    }

    /// Shared 1D IDCT butterfly used by both passes. The DESCALE
    /// const-generic differs (11 vs 18).
    #[inline(always)]
    fn idct_1d_inner<const DESCALE: u32>(
        c0: v128, c1: v128, c2: v128, c3: v128, c4: v128, c5: v128, c6: v128, c7: v128,
        c_541: v128, c_765: v128, c_1847n: v128, c_298: v128, c_1175: v128,
        c_1501: v128, c_2053: v128, c_3072: v128,
        c_899n: v128, c_2562n: v128, c_1961n: v128, c_390n: v128,
    ) -> [v128; 8] {
        // Even part — canonical form (un-refactored, matches scalar).
        let z2 = c2;
        let z3 = c6;
        let z23 = i16x8_add(z2, z3);
        let (z1_l, z1_h) = vmull_n(z23, c_541);
        // tmp2 = z1 + z3 * -F_1_847
        let (tmp2_l, tmp2_h) = vmlal_n(z1_l, z1_h, z3, c_1847n);
        // tmp3 = z1 + z2 * F_0_765
        let (tmp3_l, tmp3_h) = vmlal_n(z1_l, z1_h, z2, c_765);

        // tmp0 = (c0 + c4) << CONST_BITS  (widen via low/high extend).
        let s04 = i16x8_add(c0, c4);
        let tmp0_l = i32x4_shl(i32x4_extend_low_i16x8(s04), 13);
        let tmp0_h = i32x4_shl(i32x4_extend_high_i16x8(s04), 13);
        // tmp1 = (c0 - c4) << CONST_BITS
        let d04 = i16x8_sub(c0, c4);
        let tmp1_l = i32x4_shl(i32x4_extend_low_i16x8(d04), 13);
        let tmp1_h = i32x4_shl(i32x4_extend_high_i16x8(d04), 13);

        let tmp10_l = i32x4_add(tmp0_l, tmp3_l);
        let tmp10_h = i32x4_add(tmp0_h, tmp3_h);
        let tmp13_l = i32x4_sub(tmp0_l, tmp3_l);
        let tmp13_h = i32x4_sub(tmp0_h, tmp3_h);
        let tmp11_l = i32x4_add(tmp1_l, tmp2_l);
        let tmp11_h = i32x4_add(tmp1_h, tmp2_h);
        let tmp12_l = i32x4_sub(tmp1_l, tmp2_l);
        let tmp12_h = i32x4_sub(tmp1_h, tmp2_h);

        // Odd part.
        let t0 = c7;
        let t1 = c5;
        let t2 = c3;
        let t3 = c1;

        let z1o = i16x8_add(t0, t3);
        let z2o = i16x8_add(t1, t2);
        let z3o = i16x8_add(t0, t2);
        let z4o = i16x8_add(t1, t3);
        let z34 = i16x8_add(z3o, z4o);
        let (z5_l, z5_h) = vmull_n(z34, c_1175);

        let (t0_m_l, t0_m_h) = vmull_n(t0, c_298);
        let (t1_m_l, t1_m_h) = vmull_n(t1, c_2053);
        let (t2_m_l, t2_m_h) = vmull_n(t2, c_3072);
        let (t3_m_l, t3_m_h) = vmull_n(t3, c_1501);
        let (z1p_l, z1p_h) = vmull_n(z1o, c_899n);
        let (z2p_l, z2p_h) = vmull_n(z2o, c_2562n);
        let (z3p_l_pre, z3p_h_pre) = vmull_n(z3o, c_1961n);
        let z3p_l = i32x4_add(z3p_l_pre, z5_l);
        let z3p_h = i32x4_add(z3p_h_pre, z5_h);
        let (z4p_l_pre, z4p_h_pre) = vmull_n(z4o, c_390n);
        let z4p_l = i32x4_add(z4p_l_pre, z5_l);
        let z4p_h = i32x4_add(z4p_h_pre, z5_h);

        let t0_f_l = i32x4_add(i32x4_add(t0_m_l, z1p_l), z3p_l);
        let t0_f_h = i32x4_add(i32x4_add(t0_m_h, z1p_h), z3p_h);
        let t1_f_l = i32x4_add(i32x4_add(t1_m_l, z2p_l), z4p_l);
        let t1_f_h = i32x4_add(i32x4_add(t1_m_h, z2p_h), z4p_h);
        let t2_f_l = i32x4_add(i32x4_add(t2_m_l, z2p_l), z3p_l);
        let t2_f_h = i32x4_add(i32x4_add(t2_m_h, z2p_h), z3p_h);
        let t3_f_l = i32x4_add(i32x4_add(t3_m_l, z1p_l), z4p_l);
        let t3_f_h = i32x4_add(i32x4_add(t3_m_h, z1p_h), z4p_h);

        // Outputs — descale and narrow to i16.
        let out_0 = descale_narrow::<DESCALE>(
            i32x4_add(tmp10_l, t3_f_l),
            i32x4_add(tmp10_h, t3_f_h),
        );
        let out_1 = descale_narrow::<DESCALE>(
            i32x4_add(tmp11_l, t2_f_l),
            i32x4_add(tmp11_h, t2_f_h),
        );
        let out_2 = descale_narrow::<DESCALE>(
            i32x4_add(tmp12_l, t1_f_l),
            i32x4_add(tmp12_h, t1_f_h),
        );
        let out_3 = descale_narrow::<DESCALE>(
            i32x4_add(tmp13_l, t0_f_l),
            i32x4_add(tmp13_h, t0_f_h),
        );
        let out_4 = descale_narrow::<DESCALE>(
            i32x4_sub(tmp13_l, t0_f_l),
            i32x4_sub(tmp13_h, t0_f_h),
        );
        let out_5 = descale_narrow::<DESCALE>(
            i32x4_sub(tmp12_l, t1_f_l),
            i32x4_sub(tmp12_h, t1_f_h),
        );
        let out_6 = descale_narrow::<DESCALE>(
            i32x4_sub(tmp11_l, t2_f_l),
            i32x4_sub(tmp11_h, t2_f_h),
        );
        let out_7 = descale_narrow::<DESCALE>(
            i32x4_sub(tmp10_l, t3_f_l),
            i32x4_sub(tmp10_h, t3_f_h),
        );

        [out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7]
    }
}
