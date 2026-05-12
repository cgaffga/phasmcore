// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 2.16 (#285, 2026-05-09) — differential audit of phasm's luma
//! sub-pel MC filter against a direct transcription of ffmpeg's
//! reference C formulas in libavcodec/h264qpel_template.c.
//!
//! Goal: prove (or disprove) that phasm's `apply_luma_mv_block`
//! produces byte-identical output to ffmpeg's `put_h264_qpel*_mc??`
//! across all 16 (x_frac, y_frac) sub-pel positions for a fixed
//! reference patch + block offset.
//!
//! If divergence is found, the test prints which (x_frac, y_frac)
//! cell first diverged and where in the block. That tells us whether
//! the v1.2 §B-cascade hidden bug is in this layer.

use phasm_core::codec::h264::encoder::motion_compensation::{
    apply_luma_mv_block, apply_luma_mv_block_bipred,
};
use phasm_core::codec::h264::encoder::motion_estimation::MotionVector;
use phasm_core::codec::h264::encoder::reference_buffer::ReconFrame;

const W: u32 = 32;
const H: u32 = 32;

fn make_ref() -> ReconFrame {
    // Deterministic non-trivial luma pattern so sub-pel filters
    // produce different outputs at different (x_frac, y_frac).
    let mut y = vec![0u8; (W * H) as usize];
    for j in 0..H {
        for i in 0..W {
            let v = ((i.wrapping_mul(13) ^ j.wrapping_mul(7)).wrapping_add(j * 5 + i * 3)) & 0xFF;
            y[(j * W + i) as usize] = v as u8;
        }
    }
    let cb = vec![128u8; (W * H / 4) as usize];
    let cr = vec![128u8; (W * H / 4) as usize];
    ReconFrame { y, cb, cr, width: W, height: H, motion_grid: None }
}

/// Sample reference luma with edge replication, identical to phasm's
/// internal ref_luma_sample.
fn refsamp(rf: &ReconFrame, x: i32, y: i32) -> i32 {
    let xc = x.clamp(0, rf.width as i32 - 1) as u32;
    let yc = y.clamp(0, rf.height as i32 - 1) as u32;
    rf.y[(yc * rf.width + xc) as usize] as i32
}

fn clip255(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Direct transcription of ffmpeg's h264qpel_template.c h-lowpass at
/// (col, row): `(src[0]+src[1])*20 - (src[-1]+src[2])*5 + (src[-2]+src[3])`.
/// The "0" position corresponds to the int sample at (col, row); the
/// formula computes the half-pel sample between (col, row) and (col+1, row).
fn ff_h_lowpass(rf: &ReconFrame, col: i32, row: i32) -> i32 {
    (refsamp(rf, col, row) + refsamp(rf, col + 1, row)) * 20
        - (refsamp(rf, col - 1, row) + refsamp(rf, col + 2, row)) * 5
        + (refsamp(rf, col - 2, row) + refsamp(rf, col + 3, row))
}

/// Direct transcription of ffmpeg's v-lowpass: vertical version.
/// Result is the half-pel sample between (col, row) and (col, row+1).
fn ff_v_lowpass(rf: &ReconFrame, col: i32, row: i32) -> i32 {
    (refsamp(rf, col, row) + refsamp(rf, col, row + 1)) * 20
        - (refsamp(rf, col, row - 1) + refsamp(rf, col, row + 2)) * 5
        + (refsamp(rf, col, row - 2) + refsamp(rf, col, row + 3))
}

/// hv-lowpass: tmp = h-lowpass at row + drow for drow in [-2..3+block_h].
/// Then apply v-lowpass to those tmp values vertically.
/// Returns the j-position (center) value at (col, row), which is
/// CLIP(((sum) + 512) >> 10).
///
/// Note: ffmpeg's tmp values are STORED WITHOUT the +16 shift — they
/// are the raw weighted sums.
fn ff_hv_j(rf: &ReconFrame, col: i32, row: i32) -> i32 {
    // Sum vertically over rows {row-2, row-1, row, row+1, row+2, row+3}
    // each contributing the raw h-lowpass at that row.
    let t_b = ff_h_lowpass(rf, col, row - 2);
    let t_a = ff_h_lowpass(rf, col, row - 1);
    let t_0 = ff_h_lowpass(rf, col, row);
    let t_1 = ff_h_lowpass(rf, col, row + 1);
    let t_2 = ff_h_lowpass(rf, col, row + 2);
    let t_3 = ff_h_lowpass(rf, col, row + 3);
    (t_0 + t_1) * 20 - (t_a + t_2) * 5 + (t_b + t_3)
}

/// Reference (ffmpeg-formula) qpel sample at sub-pel position
/// (x_frac, y_frac) ∈ [0, 3]² for integer base (x_int, y_int).
///
/// The "src" pointer in ffmpeg corresponds to (x_int, y_int) here.
/// Comments reference the spec position labels (Figure 8-4) and
/// ffmpeg's `mcXY` per-position routines.
fn ff_qpel_sample(rf: &ReconFrame, x_int: i32, y_int: i32, x_frac: u8, y_frac: u8) -> u8 {
    let g = refsamp(rf, x_int, y_int);
    let h_right = refsamp(rf, x_int + 1, y_int); // "H" int sample (one to the right)
    let m_below = refsamp(rf, x_int, y_int + 1); // "M" int sample (one below)
    // half-pel intermediates clipped to [0, 255]
    let hb_at_row = |row: i32| clip255((ff_h_lowpass(rf, x_int, row) + 16) >> 5) as i32;
    let hv_at_col = |col: i32| clip255((ff_v_lowpass(rf, col, y_int) + 16) >> 5) as i32;
    let j_center = clip255((ff_hv_j(rf, x_int, y_int) + 512) >> 10) as i32;

    let val = match (x_frac, y_frac) {
        (0, 0) => g, // mc00 / G
        (1, 0) => (g + hb_at_row(y_int) + 1) >> 1, // mc10 / a
        (2, 0) => hb_at_row(y_int), // mc20 / b
        (3, 0) => (h_right + hb_at_row(y_int) + 1) >> 1, // mc30 / c
        (0, 1) => (g + hv_at_col(x_int) + 1) >> 1, // mc01 / d
        (1, 1) => (hb_at_row(y_int) + hv_at_col(x_int) + 1) >> 1, // mc11 / e
        (2, 1) => (hb_at_row(y_int) + j_center + 1) >> 1, // mc21 / f
        (3, 1) => (hb_at_row(y_int) + hv_at_col(x_int + 1) + 1) >> 1, // mc31 / g
        (0, 2) => hv_at_col(x_int), // mc02 / h
        (1, 2) => (hv_at_col(x_int) + j_center + 1) >> 1, // mc12 / i
        (2, 2) => j_center, // mc22 / j
        (3, 2) => (j_center + hv_at_col(x_int + 1) + 1) >> 1, // mc32 / k
        (0, 3) => (m_below + hv_at_col(x_int) + 1) >> 1, // mc03 / n
        (1, 3) => (hv_at_col(x_int) + hb_at_row(y_int + 1) + 1) >> 1, // mc13 / p
        (2, 3) => (j_center + hb_at_row(y_int + 1) + 1) >> 1, // mc23 / q
        (3, 3) => (hv_at_col(x_int + 1) + hb_at_row(y_int + 1) + 1) >> 1, // mc33 / r
        _ => unreachable!(),
    };
    val.clamp(0, 255) as u8
}

#[test]
fn subpel_byte_exact_audit_8x8_block() {
    // 8×8 block at (8, 8) — well-interior so edge replication doesn't fire.
    let rf = make_ref();
    let block_x = 8u32;
    let block_y = 8u32;
    let bw = 8u32;
    let bh = 8u32;

    // Iterate all 16 (x_frac, y_frac) combinations, each at integer
    // displacement zero so x_int = block_x + xl, y_int = block_y + yl.
    let mut diff_count = 0u32;
    let mut max_abs = 0i32;
    let mut first_failure: Option<(u8, u8, u32, u32, u8, u8)> = None;
    for y_frac in 0..=3 {
        for x_frac in 0..=3 {
            let mv = MotionVector { mv_x: x_frac as i16, mv_y: y_frac as i16 };
            let mut got = vec![0u8; (bw * bh) as usize];
            apply_luma_mv_block(
                &rf, block_x, block_y, bw, bh, mv,
                &mut got, bw as usize,
            );
            for yl in 0..bh {
                for xl in 0..bw {
                    let x_int = block_x as i32 + xl as i32; // mv.mv_x>>2 = 0
                    let y_int = block_y as i32 + yl as i32;
                    let want = ff_qpel_sample(&rf, x_int, y_int, x_frac, y_frac);
                    let g = got[(yl * bw + xl) as usize];
                    if g != want {
                        diff_count += 1;
                        let d = (g as i32 - want as i32).abs();
                        if d > max_abs {
                            max_abs = d;
                        }
                        if first_failure.is_none() {
                            first_failure = Some((x_frac, y_frac, xl, yl, g, want));
                        }
                    }
                }
            }
        }
    }
    if diff_count > 0 {
        let (xf, yf, xl, yl, g, w) = first_failure.unwrap();
        panic!(
            "sub-pel byte-exact audit FAILED: {} divergences, max|Δ|={}, first at \
             (x_frac={xf}, y_frac={yf}) block(xl={xl}, yl={yl}) phasm={g} ffmpeg={w}",
            diff_count, max_abs,
        );
    }
}

#[test]
fn subpel_byte_exact_audit_16x16_block() {
    // Production-size 16×16 block (entire MB) — same audit, exercises
    // the qpel16 SIMD path that's hot in B-frame MC.
    let rf = make_ref();
    let block_x = 8u32;
    let block_y = 8u32;
    let bw = 16u32;
    let bh = 16u32;

    let mut diff_count = 0u32;
    let mut max_abs = 0i32;
    let mut first_failure: Option<(u8, u8, u32, u32, u8, u8)> = None;
    for y_frac in 0..=3 {
        for x_frac in 0..=3 {
            let mv = MotionVector { mv_x: x_frac as i16, mv_y: y_frac as i16 };
            let mut got = vec![0u8; (bw * bh) as usize];
            apply_luma_mv_block(
                &rf, block_x, block_y, bw, bh, mv,
                &mut got, bw as usize,
            );
            for yl in 0..bh {
                for xl in 0..bw {
                    let x_int = block_x as i32 + xl as i32;
                    let y_int = block_y as i32 + yl as i32;
                    let want = ff_qpel_sample(&rf, x_int, y_int, x_frac, y_frac);
                    let g = got[(yl * bw + xl) as usize];
                    if g != want {
                        diff_count += 1;
                        let d = (g as i32 - want as i32).abs();
                        if d > max_abs {
                            max_abs = d;
                        }
                        if first_failure.is_none() {
                            first_failure = Some((x_frac, y_frac, xl, yl, g, want));
                        }
                    }
                }
            }
        }
    }
    if diff_count > 0 {
        let (xf, yf, xl, yl, g, w) = first_failure.unwrap();
        panic!(
            "16x16 sub-pel byte-exact audit FAILED: {} divergences, max|Δ|={}, first at \
             (x_frac={xf}, y_frac={yf}) block(xl={xl}, yl={yl}) phasm={g} ffmpeg={w}",
            diff_count, max_abs,
        );
    }
}

#[test]
fn bipred_byte_exact_audit_8x8() {
    // Bipred = average of two MC outputs, each a sub-pel MC.
    // Spec § 8.4.2.3.1: pred = (L0 + L1 + 1) >> 1.
    // ffmpeg's rnd_avg32: equivalent byte-wise to (a + b + 1) >> 1.
    // We compare phasm's `apply_luma_mv_block_bipred` against
    // taking phasm's two scalar MCs and averaging by formula. If
    // phasm's bipred internal averaging uses any divergent rounding,
    // this will catch it.
    let rf_l0 = make_ref();
    // Make L1 ref a shifted/perturbed copy so it differs from L0.
    let mut rf_l1 = make_ref();
    for v in rf_l1.y.iter_mut() {
        *v = v.wrapping_add(13);
    }

    let block_x = 8u32;
    let block_y = 8u32;
    let bw = 8u32;
    let bh = 8u32;

    let mut diff_count = 0u32;
    let mut max_abs = 0i32;

    let mvs = [
        ((0, 0), (0, 0)),
        ((1, 1), (-1, -1)),
        ((2, 0), (0, 2)),
        ((3, 3), (1, 2)),
        ((-3, 1), (2, -1)),
    ];
    for (l0, l1) in mvs {
        let mv_l0 = MotionVector { mv_x: l0.0, mv_y: l0.1 };
        let mv_l1 = MotionVector { mv_x: l1.0, mv_y: l1.1 };
        let mut got = vec![0u8; (bw * bh) as usize];
        apply_luma_mv_block_bipred(
            &rf_l0, mv_l0, &rf_l1, mv_l1,
            block_x, block_y, bw, bh,
            &mut got, bw as usize,
        );
        let mut p_l0 = vec![0u8; (bw * bh) as usize];
        let mut p_l1 = vec![0u8; (bw * bh) as usize];
        apply_luma_mv_block(&rf_l0, block_x, block_y, bw, bh, mv_l0, &mut p_l0, bw as usize);
        apply_luma_mv_block(&rf_l1, block_x, block_y, bw, bh, mv_l1, &mut p_l1, bw as usize);
        for i in 0..(bw * bh) as usize {
            let want = ((p_l0[i] as u16 + p_l1[i] as u16 + 1) >> 1) as u8;
            if got[i] != want {
                diff_count += 1;
                let d = (got[i] as i32 - want as i32).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
        }
    }
    if diff_count > 0 {
        panic!("bipred byte-exact FAILED: {} divergences, max|Δ|={}", diff_count, max_abs);
    }
}

#[test]
fn subpel_byte_exact_audit_with_negative_mvs() {
    // Re-run with non-zero integer + fractional MVs to exercise the
    // mv_int + mv_frac split logic. mv quarter-pel = -5 means int = -2,
    // frac = 3 (Rust >> 2 is arithmetic, & 3 gives 3 for negative).
    let rf = make_ref();
    let block_x = 12u32;
    let block_y = 12u32;
    let bw = 8u32;
    let bh = 8u32;

    let mut diff_count = 0u32;
    let mut max_abs = 0i32;
    let mut first_failure: Option<(i16, i16, u32, u32, u8, u8)> = None;

    let mvs = [
        (-5, -3), (-1, 0), (0, -1), (1, 2), (3, 5), (-7, -7), (5, -3),
    ];
    for (mv_x, mv_y) in mvs {
        let mv = MotionVector { mv_x, mv_y };
        let x_frac = (mv_x & 3) as u8;
        let y_frac = (mv_y & 3) as u8;
        let dx_int = mv_x as i32 >> 2;
        let dy_int = mv_y as i32 >> 2;
        let mut got = vec![0u8; (bw * bh) as usize];
        apply_luma_mv_block(
            &rf, block_x, block_y, bw, bh, mv,
            &mut got, bw as usize,
        );
        for yl in 0..bh {
            for xl in 0..bw {
                let x_int = block_x as i32 + dx_int + xl as i32;
                let y_int = block_y as i32 + dy_int + yl as i32;
                let want = ff_qpel_sample(&rf, x_int, y_int, x_frac, y_frac);
                let g = got[(yl * bw + xl) as usize];
                if g != want {
                    diff_count += 1;
                    let d = (g as i32 - want as i32).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                    if first_failure.is_none() {
                        first_failure = Some((mv_x, mv_y, xl, yl, g, want));
                    }
                }
            }
        }
    }
    if diff_count > 0 {
        let (mvx, mvy, xl, yl, g, w) = first_failure.unwrap();
        panic!(
            "sub-pel byte-exact audit (negative MVs) FAILED: {} divergences, max|Δ|={}, first at \
             mv=({mvx}, {mvy}) block(xl={xl}, yl={yl}) phasm={g} ffmpeg={w}",
            diff_count, max_abs,
        );
    }
}
