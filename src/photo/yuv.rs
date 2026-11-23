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
/// writes them (AV1 "unknown" chroma position). Rows are independent (converted in parallel with
/// the `parallel` feature; the result does not depend on it).
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
    let mut rgb = vec![0u8; w * h * 3];
    if w == 0 || h == 0 {
        return rgb;
    }
    let conv = Converter::new(layout, w, h, matrix, full_range);
    let planes = Planes { y, y_stride, u, v, c_stride };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        rgb.par_chunks_mut(w * 3).enumerate().for_each(|(yy, out)| conv.row(&planes, yy, out));
    }
    #[cfg(not(feature = "parallel"))]
    for (yy, out) in rgb.chunks_mut(w * 3).enumerate() {
        conv.row(&planes, yy, out);
    }
    rgb
}

struct Planes<'a> {
    y: &'a [u8],
    y_stride: usize,
    u: &'a [u8],
    v: &'a [u8],
    c_stride: usize,
}

/// The fixed-point constants of one conversion.
struct Converter {
    layout: Layout,
    w: usize,
    matrix: u16,
    yo: i32,
    ky: i32,
    cr_r: i32,
    cb_b: i32,
    cb_g: i32,
    cr_g: i32,
    sx: usize,
    sy: usize,
    cw: usize,
    ch: usize,
}

impl Converter {
    fn new(layout: Layout, w: usize, h: usize, matrix: u16, full_range: bool) -> Self {
        let (kr, kb) = match matrix {
            1 => (0.2126, 0.0722),
            9 => (0.2627, 0.0593),
            _ => (0.299, 0.114),
        };
        let kg = 1.0 - kr - kb;
        let (ys, cs, yo) = if full_range { (1.0, 1.0, 0) } else { (255.0 / 219.0, 255.0 / 224.0, 16) };
        let (sx, sy) = match layout {
            Layout::I420 => (1, 1),
            Layout::I422 => (1, 0),
            _ => (0, 0),
        };
        Converter {
            layout,
            w,
            matrix,
            yo,
            ky: fx(ys),
            cr_r: fx(2.0 * (1.0 - kr) * cs),
            cb_b: fx(2.0 * (1.0 - kb) * cs),
            cb_g: fx(2.0 * kb * (1.0 - kb) / kg * cs),
            cr_g: fx(2.0 * kr * (1.0 - kr) / kg * cs),
            sx,
            sy,
            cw: (w + sx) >> sx,
            ch: (h + sy) >> sy,
        }
    }

    /// Chroma at luma (x, yy): bilinear between the neighbouring samples (the sample itself for
    /// 4:4:4).
    #[inline]
    fn sample(&self, plane: &[u8], c_stride: usize, x: usize, yy: usize) -> i32 {
        if self.sx == 0 && self.sy == 0 {
            return plane[yy * c_stride + x] as i32;
        }
        // centred: sample k covers luma 2k and 2k+1; position in quarter units relative to sample 0
        let centred = |f: i32| {
            let t = f * 2 - 1;
            (t.div_euclid(4).max(0) as usize, if t < 0 { 0 } else { t.rem_euclid(4) })
        };
        let (x0, ax) = if self.sx == 1 { centred(x as i32) } else { (x, 0) };
        let (y0, ay) = if self.sy == 1 { centred(yy as i32) } else { (yy, 0) };
        let x1 = (x0 + 1).min(self.cw - 1);
        let y1 = (y0 + 1).min(self.ch - 1);
        let x0 = x0.min(self.cw - 1);
        let y0 = y0.min(self.ch - 1);
        let p = |xx: usize, yy: usize| plane[yy * c_stride + xx] as i32;
        let wx = ax;
        let top = p(x0, y0) * (4 - wx) + p(x1, y0) * wx;
        let bot = p(x0, y1) * (4 - wx) + p(x1, y1) * wx;
        (top * (4 - ay) + bot * ay + 8) >> 4
    }

    /// Row `yy` into `out` (`w * 3` bytes).
    fn row(&self, pl: &Planes<'_>, yy: usize, out: &mut [u8]) {
        let ys = &pl.y[yy * pl.y_stride..yy * pl.y_stride + self.w];
        for (x, (o, &luma)) in out.chunks_exact_mut(3).zip(ys).enumerate() {
            let l = (luma as i32 - self.yo) * self.ky;
            if self.layout == Layout::Mono {
                let g = clamp8((l + HALF) >> 16);
                o.copy_from_slice(&[g, g, g]);
                continue;
            }
            if self.matrix == 0 {
                // identity: Y = G, U = B, V = R
                let (b, r) = (self.sample(pl.u, pl.c_stride, x, yy), self.sample(pl.v, pl.c_stride, x, yy));
                o.copy_from_slice(&[clamp8(r), luma, clamp8(b)]);
                continue;
            }
            let cb = self.sample(pl.u, pl.c_stride, x, yy) - 128;
            let cr = self.sample(pl.v, pl.c_stride, x, yy) - 128;
            o[0] = clamp8((l + self.cr_r * cr + HALF) >> 16);
            o[1] = clamp8((l - self.cb_g * cb - self.cr_g * cr + HALF) >> 16);
            o[2] = clamp8((l + self.cb_b * cb + HALF) >> 16);
        }
    }
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
