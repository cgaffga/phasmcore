// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! RGB <-> YCbCr for photo attachments, in integers (identical on every target).

/// 16-bit fixed point.
const FIX: i32 = 1 << 16;
const HALF: i32 = 1 << 15;

const fn fx(v: f64) -> i32 {
    (v * FIX as f64 + if v < 0.0 { -0.5 } else { 0.5 }) as i32
}

#[inline]
fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Interleaved 8-bit RGB (`w` x `h`, both even) -> 4:2:0 planes, BT.601 full range (the JPEG
/// matrix). Chroma is the average of each 2x2 block.
pub fn rgb_to_i420(rgb: &[u8], w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    debug_assert!(w.is_multiple_of(2) && h.is_multiple_of(2) && rgb.len() >= w * h * 3);
    let mut y = vec![0u8; w * h];
    for (i, p) in rgb.chunks_exact(3).take(w * h).enumerate() {
        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
        y[i] = clamp8((fx(0.299) * r + fx(0.587) * g + fx(0.114) * b + HALF) >> 16);
    }
    let (cw, ch) = (w / 2, h / 2);
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for cy in 0..ch {
        for cx in 0..cw {
            let (mut r, mut g, mut b) = (0, 0, 0);
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let p = ((2 * cy + dy) * w + 2 * cx + dx) * 3;
                r += rgb[p] as i32;
                g += rgb[p + 1] as i32;
                b += rgb[p + 2] as i32;
            }
            // sums of 4: the >> 18 divides by 4 * FIX
            let cb = fx(-0.168736) * r + fx(-0.331264) * g + fx(0.5) * b;
            let cr = fx(0.5) * r + fx(-0.418688) * g + fx(-0.081312) * b;
            u[cy * cw + cx] = clamp8(128 + ((cb + (1 << 17)) >> 18));
            v[cy * cw + cx] = clamp8(128 + ((cr + (1 << 17)) >> 18));
        }
    }
    (y, u, v)
}

/// The layout of decoded planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Mono,
    I420,
    I422,
    I444,
}

/// Decoded planes to interleaved RGB. `matrix` is the ISO/IEC 23091-2 code point: 0 = identity
/// (GBR), 1 = BT.709, 9 = BT.2020 (non-constant), anything else is treated as BT.601. Chroma is
/// upsampled bilinearly from samples centred on their 2x2 (or 2x1) luma block, as `rgb_to_i420`
/// writes them (AV1 "unknown" chroma position).
#[allow(clippy::too_many_arguments)]
pub fn yuv_to_rgb(
    layout: Layout,
    w: usize,
    h: usize,
    y: &[u8],
    y_stride: usize,
    u: &[u8],
    v: &[u8],
    c_stride: usize,
    matrix: u16,
    full_range: bool,
) -> Vec<u8> {
    let (kr, kb) = match matrix {
        1 => (0.2126, 0.0722),
        9 => (0.2627, 0.0593),
        _ => (0.299, 0.114),
    };
    let kg = 1.0 - kr - kb;
    let (ys, cs, yo) = if full_range { (1.0, 1.0, 0) } else { (255.0 / 219.0, 255.0 / 224.0, 16) };
    let ky = fx(ys);
    let cr_r = fx(2.0 * (1.0 - kr) * cs);
    let cb_b = fx(2.0 * (1.0 - kb) * cs);
    let cb_g = fx(2.0 * kb * (1.0 - kb) / kg * cs);
    let cr_g = fx(2.0 * kr * (1.0 - kr) / kg * cs);
    let (sx, sy) = match layout {
        Layout::I420 => (1, 1),
        Layout::I422 => (1, 0),
        _ => (0, 0),
    };
    let (cw, ch) = ((w + sx) >> sx, (h + sy) >> sy);
    // chroma at (x, y) in 1/2-sample units: bilinear between the neighbouring samples
    let sample = |plane: &[u8], x: usize, yy: usize| -> i32 {
        let (fx2, fy2) = (x as i32, yy as i32);
        // centred: sample k covers luma 2k and 2k+1; position in quarter units relative to sample 0
        let centred = |f: i32| {
            let t = f * 2 - 1;
            (t.div_euclid(4).max(0) as usize, if t < 0 { 0 } else { t.rem_euclid(4) })
        };
        let (x0, ax) = if sx == 1 { centred(fx2) } else { (x, 0) };
        let (y0, ay) = if sy == 1 { centred(fy2) } else { (yy, 0) };
        let x1 = (x0 + 1).min(cw - 1);
        let y1 = (y0 + 1).min(ch - 1);
        let x0 = x0.min(cw - 1);
        let y0 = y0.min(ch - 1);
        let p = |xx: usize, yy: usize| plane[yy * c_stride + xx] as i32;
        let wx = ax;
        let top = p(x0, y0) * (4 - wx) + p(x1, y0) * wx;
        let bot = p(x0, y1) * (4 - wx) + p(x1, y1) * wx;
        (top * (4 - ay) + bot * ay + 8) >> 4
    };
    let mut rgb = vec![0u8; w * h * 3];
    for yy in 0..h {
        for x in 0..w {
            let l = (y[yy * y_stride + x] as i32 - yo) * ky;
            let o = (yy * w + x) * 3;
            if layout == Layout::Mono {
                let g = clamp8((l + HALF) >> 16);
                rgb[o..o + 3].copy_from_slice(&[g, g, g]);
                continue;
            }
            if matrix == 0 {
                // identity: Y = G, U = B, V = R
                let (g, b, r) = (y[yy * y_stride + x], sample(u, x, yy), sample(v, x, yy));
                rgb[o..o + 3].copy_from_slice(&[clamp8(r), g, clamp8(b)]);
                continue;
            }
            let cb = sample(u, x, yy) - 128;
            let cr = sample(v, x, yy) - 128;
            rgb[o] = clamp8((l + cr_r * cr + HALF) >> 16);
            rgb[o + 1] = clamp8((l - cb_g * cb - cr_g * cr + HALF) >> 16);
            rgb[o + 2] = clamp8((l + cb_b * cb + HALF) >> 16);
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grey_and_primaries_round_trip() {
        let (w, h) = (4, 2);
        for c in [[128u8, 128, 128], [255, 0, 0], [0, 255, 0], [0, 0, 255], [10, 200, 90]] {
            let rgb: Vec<u8> = c.iter().copied().cycle().take(w * h * 3).collect();
            let (y, u, v) = rgb_to_i420(&rgb, w, h);
            let back = yuv_to_rgb(Layout::I420, w, h, &y, w, &u, &v, w / 2, 6, true);
            for (a, b) in rgb.iter().zip(&back) {
                assert!((*a as i32 - *b as i32).abs() <= 2, "{c:?}: {a} vs {b}");
            }
        }
    }
}
