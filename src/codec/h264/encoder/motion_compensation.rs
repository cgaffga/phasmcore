// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Motion compensation per H.264 § 8.4.2.2. Phase 6B.1.
//!
//! - Luma: 6-tap quarter-pel filter (§ 8.4.2.2.1, Figure 8-4,
//!   Table 8-12).
//! - Chroma: bilinear eighth-pel filter (§ 8.4.2.2.2, Eq. 8-272).
//!
//! Both clip integer sample coordinates to the reference frame's
//! edges via `Clip3(0, width−1, …)` — replicating edge pixels for
//! out-of-bounds MVs.
//!
//! Algorithm note: docs/design/h264-encoder-algorithms/motion-compensation.md

use super::motion_estimation::MotionVector;
use super::reference_buffer::ReconFrame;

/// 6-tap qpel filter kernel per spec Eq. 8-243 / 8-244.
/// Applied to 6 samples, output = sum / 32 (with +16 rounding).
const LUMA_FILTER: [i32; 6] = [1, -5, 20, 20, -5, 1];

/// Apply the 6-tap filter to 6 i32 samples and return the filtered
/// half-sample intermediate `(s_0, s_1, s_2, s_3, s_4, s_5)` →
/// `b1 = s0 - 5*s1 + 20*s2 + 20*s3 - 5*s4 + s5`.
#[inline]
fn filter6(s: [i32; 6]) -> i32 {
    s[0] * LUMA_FILTER[0]
        + s[1] * LUMA_FILTER[1]
        + s[2] * LUMA_FILTER[2]
        + s[3] * LUMA_FILTER[3]
        + s[4] * LUMA_FILTER[4]
        + s[5] * LUMA_FILTER[5]
}

/// Clip a luma sample to [0, 255].
#[inline]
fn clip1y(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Fetch a luma sample from the reference frame with edge replication.
#[inline]
fn ref_luma_sample(reference: &ReconFrame, x: i32, y: i32) -> i32 {
    let xc = x.clamp(0, reference.width as i32 - 1) as u32;
    let yc = y.clamp(0, reference.height as i32 - 1) as u32;
    reference.y_at(xc, yc) as i32
}

/// Apply a luma MV to fill a `block_w × block_h` prediction
/// rectangle at `(block_x, block_y)` in the current frame.
///
/// Samples are written into `out` at `out_stride`-spaced rows; the
/// caller can point into a larger buffer (e.g. a 16×16 MB) and paint
/// per-partition sub-rectangles in place.
///
/// `mv` is in quarter-pel units. Integer part = `mv >> 2`, fractional
/// phase = `mv & 3`.
pub fn apply_luma_mv_block(
    reference: &ReconFrame,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    mv: MotionVector,
    out: &mut [u8],
    out_stride: usize,
) {
    if super::simd::apply_luma_mv_block_dispatch(
        &reference.y, reference.width, reference.height,
        block_x, block_y, block_w, block_h,
        mv.mv_x, mv.mv_y,
        out, out_stride,
    ) {
        return;
    }
    apply_luma_mv_block_scalar(reference, block_x, block_y, block_w, block_h, mv, out, out_stride);
}

#[inline]
fn apply_luma_mv_block_scalar(
    reference: &ReconFrame,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    mv: MotionVector,
    out: &mut [u8],
    out_stride: usize,
) {
    for yl in 0..block_h {
        for xl in 0..block_w {
            let x_int = block_x as i32 + (mv.mv_x as i32 >> 2) + xl as i32;
            let y_int = block_y as i32 + (mv.mv_y as i32 >> 2) + yl as i32;
            let x_frac = (mv.mv_x & 3) as u8;
            let y_frac = (mv.mv_y & 3) as u8;
            out[(yl as usize) * out_stride + xl as usize] =
                sample_luma_frac(reference, x_int, y_int, x_frac, y_frac);
        }
    }
}

/// Thin 16×16 wrapper around [`apply_luma_mv_block`]. Kept so the
/// pre-6B.3.2 call sites don't have to change immediately.
pub fn apply_luma_mv_16x16(
    reference: &ReconFrame,
    block_x: u32,
    block_y: u32,
    mv: MotionVector,
) -> [[u8; 16]; 16] {
    let mut out = [[0u8; 16]; 16];
    apply_luma_mv_block(
        reference,
        block_x,
        block_y,
        16,
        16,
        mv,
        out.as_flattened_mut(),
        16,
    );
    out
}

/// Compute a single fractional-pel luma sample at `(x_int, y_int)`
/// with fractional offset `(x_frac, y_frac)` where frac ∈ [0, 3].
///
/// Matches spec Figure 8-4 / Table 8-12 position labels:
///   (0, 0) → G (integer sample)
///   (1, 0) → a = (G + b + 1) >> 1
///   (2, 0) → b = Clip1y((b1 + 16) >> 5)
///   ... (16 total labels).
fn sample_luma_frac(
    reference: &ReconFrame,
    x_int: i32,
    y_int: i32,
    x_frac: u8,
    y_frac: u8,
) -> u8 {
    // Fast path: integer sample.
    if x_frac == 0 && y_frac == 0 {
        return ref_luma_sample(reference, x_int, y_int) as u8;
    }

    // Helper closures for the various intermediate values.
    // Positions in Figure 8-4 (relative to G = (0, 0) int):
    //   G = (0, 0), H = (1, 0), M = (0, 1), N = (1, 1).
    let ref_g = |dx: i32, dy: i32| ref_luma_sample(reference, x_int + dx, y_int + dy);

    // Horizontal half-pel intermediates `b1` at integer rows.
    // For a row at vertical offset dy, b1 at horizontal int offset (0) is:
    //   b1 = E − 5F + 20G + 20H − 5I + J
    //   where E..J are at (-2, dy), (-1, dy), (0, dy), (1, dy), (2, dy), (3, dy).
    //
    // Bound: each ref_g sample ∈ [0, 255]. Worst-case magnitude is
    // |b1| ≤ (1 + 5 + 20 + 20 + 5 + 1) · 255 = 13260 → fits i16 (max 32767)
    // with margin. Returning i16 here doubles NEON `int16x8_t` lane density
    // (8 lanes/128-bit reg) over the i32 alternative (4 lanes), the only
    // reason this isn't just `i32`. Vertical-pass `j1` reads multiple b1's
    // and CAN'T fit in i16 (worst case 6×13260·sum-abs ≈ 689k → must stay
    // i32) so consumers cast back to i32.
    let b1 = |dy: i32| -> i16 {
        filter6([
            ref_g(-2, dy),
            ref_g(-1, dy),
            ref_g(0, dy),
            ref_g(1, dy),
            ref_g(2, dy),
            ref_g(3, dy),
        ]) as i16
    };
    // Vertical half-pel intermediate `h1` at integer column dx:
    //   h1 = A − 5C + 20G + 20M − 5R + T
    //   where A..T are at (dx, -2), (dx, -1), (dx, 0), (dx, 1), (dx, 2), (dx, 3).
    // Same i16 bound as b1.
    let h1 = |dx: i32| -> i16 {
        filter6([
            ref_g(dx, -2),
            ref_g(dx, -1),
            ref_g(dx, 0),
            ref_g(dx, 1),
            ref_g(dx, 2),
            ref_g(dx, 3),
        ]) as i16
    };
    // Final half-pel values (post-clip). Cast back to i32 for arithmetic.
    let b = |dy: i32| clip1y((b1(dy) as i32 + 16) >> 5) as i32;
    let h = |dx: i32| clip1y((h1(dx) as i32 + 16) >> 5) as i32;
    // `j` is the center quarter-pel: apply 6-tap filter vertically to
    // a column of horizontally-pre-filtered `b1` values. Per
    // Eq. 8-248: j1 = aa − 5bb + 20·b1 + 20·s1 − 5·gg + hh.
    // where aa, bb, gg, s1, hh are b1 at dy = -2, -1, 1, 2, 3 relative
    // to the target row (dy=0). j1 needs i32 — sum of 6 i16 b1 values
    // weighted by sum-abs 52 exceeds i16 range.
    let j = || {
        let j1 = filter6([
            b1(-2) as i32,
            b1(-1) as i32,
            b1(0) as i32,
            b1(1) as i32,
            b1(2) as i32,
            b1(3) as i32,
        ]);
        clip1y((j1 + 512) >> 10) as i32
    };

    // Integer sample shortcuts.
    let g_sample = ref_g(0, 0);
    let h_int = ref_g(1, 0); // "H" in Figure 8-4 = integer sample at (1, 0)
    let m_int = ref_g(0, 1); // "M" = (0, 1)

    // Per Table 8-12, (x_frac, y_frac) → position label → value.
    let val = match (x_frac, y_frac) {
        (0, 0) => g_sample, // G
        (1, 0) => (g_sample + b(0) + 1) >> 1, // a
        (2, 0) => b(0), // b
        (3, 0) => (h_int + b(0) + 1) >> 1, // c
        (0, 1) => (g_sample + h(0) + 1) >> 1, // d
        (1, 1) => (b(0) + h(0) + 1) >> 1, // e
        (2, 1) => (b(0) + j() + 1) >> 1, // f
        (3, 1) => (b(0) + h(1) + 1) >> 1, // g  — "m" in Fig 8-4 is h at dx=1
        (0, 2) => h(0), // h
        (1, 2) => (h(0) + j() + 1) >> 1, // i
        (2, 2) => j(), // j
        (3, 2) => (j() + h(1) + 1) >> 1, // k
        (0, 3) => (m_int + h(0) + 1) >> 1, // n
        (1, 3) => (h(0) + b(1) + 1) >> 1, // p  — "s" in Fig 8-4 is b at dy=1
        (2, 3) => (j() + b(1) + 1) >> 1, // q
        (3, 3) => (h(1) + b(1) + 1) >> 1, // r
        _ => unreachable!("x_frac/y_frac in [0, 3]"),
    };
    val.clamp(0, 255) as u8
}

/// Fetch a chroma sample from the reference frame with edge replication.
#[inline]
fn ref_chroma_sample(
    reference: &ReconFrame,
    component: u8,
    x: i32,
    y: i32,
) -> i32 {
    let c_width = reference.width / 2;
    let c_height = reference.height / 2;
    let xc = x.clamp(0, c_width as i32 - 1) as u32;
    let yc = y.clamp(0, c_height as i32 - 1) as u32;
    reference.chroma_at(component, xc, yc) as i32
}

/// Apply a chroma MV to fill a `block_w × block_h` chroma prediction
/// rectangle at chroma-sample coordinates `(block_x, block_y)`. `mv`
/// is in QUARTER-PEL LUMA units; spec Eq. 8-229/8-230 converts to
/// eighth-pel chroma internally.
pub fn apply_chroma_mv_block(
    reference: &ReconFrame,
    component: u8,
    block_x: u32,
    block_y: u32,
    block_w: u32,
    block_h: u32,
    mv: MotionVector,
    out: &mut [u8],
    out_stride: usize,
) {
    for yc in 0..block_h {
        for xc in 0..block_w {
            let x_int = block_x as i32 + (mv.mv_x as i32 >> 3) + xc as i32;
            let y_int = block_y as i32 + (mv.mv_y as i32 >> 3) + yc as i32;
            let x_frac = (mv.mv_x & 7) as i32;
            let y_frac = (mv.mv_y & 7) as i32;

            let a = ref_chroma_sample(reference, component, x_int, y_int);
            let b = ref_chroma_sample(reference, component, x_int + 1, y_int);
            let c = ref_chroma_sample(reference, component, x_int, y_int + 1);
            let d = ref_chroma_sample(reference, component, x_int + 1, y_int + 1);

            let v = ((8 - x_frac) * (8 - y_frac) * a
                + x_frac * (8 - y_frac) * b
                + (8 - x_frac) * y_frac * c
                + x_frac * y_frac * d
                + 32)
                >> 6;
            out[(yc as usize) * out_stride + xc as usize] = v.clamp(0, 255) as u8;
        }
    }
}

/// Thin 8×8 wrapper around [`apply_chroma_mv_block`]. Kept for
/// pre-6B.3.2 call sites.
pub fn apply_chroma_mv_8x8(
    reference: &ReconFrame,
    component: u8,
    block_x: u32,
    block_y: u32,
    mv: MotionVector,
) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    apply_chroma_mv_block(
        reference,
        component,
        block_x,
        block_y,
        8,
        8,
        mv,
        out.as_flattened_mut(),
        8,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::reconstruction::ReconBuffer;

    fn make_reference(width: u32, height: u32, fill_y: impl Fn(u32, u32) -> u8) -> ReconFrame {
        let mut rb = ReconBuffer::new(width, height).unwrap();
        for y in 0..height {
            for x in 0..width {
                rb.y[(y * width + x) as usize] = fill_y(x, y);
            }
        }
        // Neutral chroma = 128 everywhere.
        for v in rb.cb.iter_mut() { *v = 128; }
        for v in rb.cr.iter_mut() { *v = 128; }
        ReconFrame::snapshot(&rb)
    }

    #[test]
    fn integer_mv_zero_returns_ref_block_unchanged() {
        let reference = make_reference(64, 48, |x, y| ((x * 3 + y * 5) & 0xFF) as u8);
        let mv = MotionVector { mv_x: 0, mv_y: 0 };
        let block = apply_luma_mv_16x16(&reference, 16, 16, mv);
        for dy in 0..16 {
            for dx in 0..16 {
                let expected = ((16 + dx) * 3 + (16 + dy) * 5) & 0xFF;
                assert_eq!(
                    block[dy as usize][dx as usize],
                    expected as u8,
                    "pixel ({dx}, {dy}) mismatch"
                );
            }
        }
    }

    #[test]
    fn integer_mv_nonzero_shifts_block() {
        // Reference is a horizontal gradient. mv = (4, 0) in qpel =
        // 1 integer pel right. The extracted block at (16, 16) should
        // equal the reference block at (17, 16).
        let reference = make_reference(64, 48, |x, _y| x as u8);
        let mv = MotionVector { mv_x: 4, mv_y: 0 }; // 1-int-pel right
        let block = apply_luma_mv_16x16(&reference, 16, 16, mv);
        for dy in 0..16 {
            for dx in 0..16 {
                assert_eq!(
                    block[dy][dx] as u32,
                    (16 + dx as u32 + 1),
                    "pixel ({dx}, {dy})"
                );
            }
        }
    }

    #[test]
    fn flat_reference_qpel_returns_flat() {
        // Flat reference → any MC'd block = flat same value.
        let reference = make_reference(64, 48, |_, _| 100);
        for (mvx, mvy) in [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (2, 2), (3, 3)] {
            let mv = MotionVector { mv_x: mvx, mv_y: mvy };
            let block = apply_luma_mv_16x16(&reference, 16, 16, mv);
            for row in &block {
                for &v in row {
                    assert_eq!(v, 100, "flat expected for mv=({mvx},{mvy}), got {v}");
                }
            }
        }
    }

    #[test]
    fn horizontal_halfpel_applies_filter() {
        // Delta reference: sample value at x=20, y=16 is 255, else 128.
        // With mv (2, 0) = half-pel right, the filter should blend.
        let reference = make_reference(
            64, 48,
            |x, y| if x == 20 && y == 16 { 255 } else { 128 },
        );
        let mv = MotionVector { mv_x: 2, mv_y: 0 }; // half-pel right
        let block = apply_luma_mv_16x16(&reference, 16, 16, mv);
        // Pixel (3, 0) in the output → reference pixel (16+3, 16) = int (19, 16)
        // with half-pel right. 6-tap kernel centered between 19 and 20:
        //   b1 = s[17] - 5*s[18] + 20*s[19] + 20*s[20] - 5*s[21] + s[22]
        //      = 128 - 5*128 + 20*128 + 20*255 - 5*128 + 128
        //      = 128(1 - 5 + 20 - 5 + 1) + 20*255
        //      = 128*12 + 5100 = 1536 + 5100 = 6636
        //   b  = clip1y((6636 + 16) >> 5) = 6652 >> 5 = 207.
        // Pixel value should be 207.
        assert_eq!(block[0][3], 207);
    }

    #[test]
    fn edge_clipping_at_top_left() {
        // MV that pulls integer part outside the frame; should clip
        // without panic.
        let reference = make_reference(64, 48, |_, _| 100);
        let mv = MotionVector { mv_x: -80, mv_y: -80 }; // -20 int pels
        let block = apply_luma_mv_16x16(&reference, 16, 16, mv);
        // Edge replication of a flat frame is still flat.
        for row in &block {
            for &v in row {
                assert_eq!(v, 100);
            }
        }
    }

    #[test]
    fn chroma_integer_mv_returns_ref_unchanged() {
        let _reference = make_reference(64, 48, |_, _| 128);
        // Custom chroma: Cb = some pattern.
        let mut rb = ReconBuffer::new(64, 48).unwrap();
        for y in 0..24 {
            for x in 0..32 {
                rb.cb[(y * 32 + x) as usize] = ((x * 2) & 0xFF) as u8;
            }
        }
        let reference2 = ReconFrame::snapshot(&rb);
        let mv = MotionVector { mv_x: 0, mv_y: 0 };
        let block = apply_chroma_mv_8x8(&reference2, 0, 8, 8, mv);
        for dy in 0..8 {
            for dx in 0..8 {
                assert_eq!(
                    block[dy][dx] as u32,
                    (8 + dx as u32) * 2 & 0xFF,
                    "Cb pixel ({dx}, {dy})"
                );
            }
        }
    }

    #[test]
    fn chroma_bilinear_produces_blended() {
        // Chroma delta: Cb at (4, 4) is 240, else 128.
        let mut rb = ReconBuffer::new(64, 48).unwrap();
        for v in rb.y.iter_mut() { *v = 128; }
        for v in rb.cb.iter_mut() { *v = 128; }
        rb.cb[(4 * 32 + 4) as usize] = 240;
        for v in rb.cr.iter_mut() { *v = 128; }
        let reference = ReconFrame::snapshot(&rb);

        // mv = (4, 4) qpel luma = eighth-pel chroma (4, 4).
        // xInt = block_x + (mv_x >> 3) + xc. xFrac = mv_x & 7 = 4.
        // For output pixel (xc=4, yc=4) with block_x=0, block_y=0:
        //   x_int = 0 + 0 + 4 = 4. y_int = 4.
        // So we sample at chroma (4.5, 4.5). A/B/C/D at (4, 4), (5, 4),
        // (4, 5), (5, 5). Our delta at (4, 4) = 240, others = 128.
        // pred = ((4)(4)·240 + (4)(4)·128 + (4)(4)·128 + (4)(4)·128 + 32) >> 6
        //      = (3840 + 2048 + 2048 + 2048 + 32) >> 6
        //      = 10016 >> 6 = 156.
        let mv = MotionVector { mv_x: 4, mv_y: 4 };
        let block = apply_chroma_mv_8x8(&reference, 0, 0, 0, mv);
        assert_eq!(block[4][4], 156);
    }
}
