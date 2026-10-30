// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Baseline JPEG writer for Deep Cover covers (libjpeg-turbo compatible).
//!
//! Deep Cover (side-informed Ghost) needs the *pre-rounding* DCT
//! coefficients of the cover JPEG. Those are only known exactly when the
//! encoder that produced the cover is reproduced bit for bit, so phasm
//! writes the cover itself. The writer reproduces what `cjpeg -quality Q`
//! (libjpeg-turbo, default settings) writes for an RGB input:
//!
//! - colour conversion `rgb_ycc_convert` (16-bit fixed point, `jccolor.c`),
//! - 4:2:0 chroma, `h2v2_downsample` with the 1,2,1,2 bias (`jcsample.c`),
//!   edge replication as `expand_right_edge` / `expand_bottom_edge`,
//! - islow forward DCT (`jfdctint.c`, [`fdct_islow`]),
//! - quantization by the 16-bit reciprocal method of `jcdctmgr.c`
//!   (`compute_reciprocal` + `quantize`, what the SIMD builds run),
//! - IJG (Annex K) quantization tables scaled by `jpeg_quality_scaling`,
//!   forced to baseline (≤ 255),
//! - dummy blocks at the right/bottom MCU edge as `jccoefct.c` makes them
//!   (AC zero, DC copied),
//! - Annex K Huffman tables (no `-optimize`), JFIF 1.01 APP0, one DQT / DHT
//!   marker per table, in libjpeg's order.
//!
//! The output is byte-identical to libjpeg-turbo 3.1 `cjpeg -quality Q` on
//! the same pixels (tested on BOSSbase and random images), so the cover
//! carries the most common encoder fingerprint there is.
//!
//! Integer arithmetic only: identical on every architecture and on WASM.

use super::dct::DctGrid;
use super::error::Result;
use super::frame::parse_sof;
use super::pixels_aan::fdct_islow;
use super::scan::{encode_scan, ScanComponent};
use super::tables::HuffmanSpec;
use super::zigzag::NATURAL_TO_ZIGZAG;

/// IJG / Annex K luminance quantization table (natural order).
pub const STD_LUMA_QT: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, //
    12, 12, 14, 19, 26, 58, 60, 55, //
    14, 13, 16, 24, 40, 57, 69, 56, //
    14, 17, 22, 29, 51, 87, 80, 62, //
    18, 22, 37, 56, 68, 109, 103, 77, //
    24, 35, 55, 64, 81, 104, 113, 92, //
    49, 64, 78, 87, 103, 121, 120, 101, //
    72, 92, 95, 98, 112, 100, 103, 99,
];

/// IJG / Annex K chrominance quantization table (natural order).
pub const STD_CHROMA_QT: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, //
    18, 21, 26, 66, 99, 99, 99, 99, //
    24, 26, 56, 99, 99, 99, 99, 99, //
    47, 66, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99,
];

const DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const AC_LUMA_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61,
    0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52,
    0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25,
    0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3,
    0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8,
    0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];
const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_CHROMA_VALS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61,
    0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33,
    0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18,
    0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
    0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63,
    0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a,
    0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
    0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
    0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
];

/// `jpeg_quality_scaling` (jcparam.c).
pub fn quality_scaling(quality: u8) -> u32 {
    let q = quality.clamp(1, 100) as u32;
    if q < 50 { 5000 / q } else { 200 - 2 * q }
}

/// `jpeg_add_quant_table` with `force_baseline = TRUE`.
pub fn scaled_quant_table(base: &[u16; 64], quality: u8) -> [u16; 64] {
    let scale = quality_scaling(quality);
    let mut out = [0u16; 64];
    for i in 0..64 {
        out[i] = ((base[i] as u32 * scale + 50) / 100).clamp(1, 255) as u16;
    }
    out
}

/// libjpeg-turbo's quantizer for one coefficient: the 16-bit reciprocal
/// method (`compute_reciprocal` with `sizeof(DCTELEM) == 2`, then
/// `quantize`). Precomputed per table entry.
#[derive(Clone, Copy)]
struct Recip {
    fq: u32,
    corr: u32,
    r: u32,
}

impl Recip {
    fn new(divisor: u32) -> Self {
        // divisor = quantval << 3 ≥ 8, so the divisor == 1 special case
        // of compute_reciprocal never applies.
        let b = 31 - divisor.leading_zeros(); // flss(divisor) − 1
        let mut r = 16 + b;
        let mut fq = (1u32 << r) / divisor;
        let fr = (1u32 << r) % divisor;
        let mut c = divisor / 2;
        if fr == 0 {
            fq >>= 1;
            r -= 1;
        } else if fr <= divisor / 2 {
            c += 1;
        } else {
            fq += 1;
        }
        Recip { fq, corr: c, r }
    }

    #[inline]
    fn quantize(self, v: i16) -> i16 {
        let a = (v as i32).unsigned_abs();
        let q = (((a + self.corr) & 0xFFFF) * self.fq) >> self.r;
        if v < 0 { -(q as i32) as i16 } else { q as i16 }
    }
}

/// Divisors (quantval × 8, the islow scale) and their reciprocals.
fn recips(qt: &[u16; 64]) -> [Recip; 64] {
    let mut out = [Recip { fq: 0, corr: 0, r: 0 }; 64];
    for i in 0..64 {
        out[i] = Recip::new((qt[i] as u32) << 3);
    }
    out
}

// rgb_ycc_convert constants (jccolor.c): FIX(x) = x·2¹⁶ + ½.
const SCALEBITS: u32 = 16;
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);
const CBCR_OFFSET: i32 = 128 << SCALEBITS;

#[inline]
fn y_of(r: i32, g: i32, b: i32) -> i32 {
    (19595 * r + 38470 * g + 7471 * b + ONE_HALF) >> SCALEBITS
}

#[inline]
fn cb_of(r: i32, g: i32, b: i32) -> i32 {
    (-11059 * r - 21709 * g + 32768 * b + CBCR_OFFSET + ONE_HALF - 1) >> SCALEBITS
}

#[inline]
fn cr_of(r: i32, g: i32, b: i32) -> i32 {
    (32768 * r - 27439 * g - 5329 * b + CBCR_OFFSET + ONE_HALF - 1) >> SCALEBITS
}

/// Geometry of a 4:2:0 baseline frame for `w × h` pixels.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub width: usize,
    pub height: usize,
    pub mcus_wide: usize,
    pub mcus_tall: usize,
    /// Luma blocks holding image data (`ceil(w/8)`, `ceil(h/8)`); the luma
    /// grid is `2·mcus_wide × 2·mcus_tall`, the rest are dummy blocks.
    pub y_wib: usize,
    pub y_hib: usize,
}

impl Geometry {
    pub fn new(width: usize, height: usize) -> Self {
        Geometry {
            width,
            height,
            mcus_wide: width.div_ceil(16),
            mcus_tall: height.div_ceil(16),
            y_wib: width.div_ceil(8),
            y_hib: height.div_ceil(8),
        }
    }

    pub fn y_blocks_wide(&self) -> usize {
        self.mcus_wide * 2
    }

    pub fn y_blocks_tall(&self) -> usize {
        self.mcus_tall * 2
    }

    /// True when luma block (br, bc) holds image data (not a dummy block).
    pub fn y_block_is_real(&self, br: usize, bc: usize) -> bool {
        br < self.y_hib && bc < self.y_wib
    }
}

/// Level-shifted luma samples of block (br, bc), edge-replicated.
#[inline]
fn luma_block(rgb: &[u8], g: &Geometry, br: usize, bc: usize) -> [i16; 64] {
    let mut d = [0i16; 64];
    for row in 0..8 {
        let y = (br * 8 + row).min(g.height - 1);
        let line = &rgb[y * g.width * 3..(y + 1) * g.width * 3];
        for col in 0..8 {
            let x = (bc * 8 + col).min(g.width - 1);
            let p = &line[x * 3..x * 3 + 3];
            d[row * 8 + col] = (y_of(p[0] as i32, p[1] as i32, p[2] as i32) - 128) as i16;
        }
    }
    d
}

/// Level-shifted chroma samples (Cb if `cb`, else Cr) of chroma block
/// (br, bc): `h2v2_downsample` of the edge-replicated full-size plane.
#[inline]
fn chroma_block(rgb: &[u8], g: &Geometry, br: usize, bc: usize, cb: bool) -> [i16; 64] {
    let ch = g.height.div_ceil(2); // downsampled rows holding data
    let conv = |x: usize, y: usize| -> i32 {
        let i = (y * g.width + x) * 3;
        let (r, gg, b) = (rgb[i] as i32, rgb[i + 1] as i32, rgb[i + 2] as i32);
        if cb { cb_of(r, gg, b) } else { cr_of(r, gg, b) }
    };
    let mut d = [0i16; 64];
    for row in 0..8 {
        let j = (br * 8 + row).min(ch - 1);
        let y0 = 2 * j;
        let y1 = (2 * j + 1).min(g.height - 1);
        for col in 0..8 {
            let k = bc * 8 + col;
            let x0 = (2 * k).min(g.width - 1);
            let x1 = (2 * k + 1).min(g.width - 1);
            let bias = if k & 1 == 0 { 1 } else { 2 };
            let s = (conv(x0, y0) + conv(x1, y0) + conv(x0, y1) + conv(x1, y1) + bias) >> 2;
            d[row * 8 + col] = (s - 128) as i16;
        }
    }
    d
}

/// Luma of every real block: islow output before quantization (`raw`,
/// libjpeg scale) and the quantized coefficients (`quant`), natural order.
/// Calls `f(br, bc, &raw, &quant)` in raster order while it returns `true`
/// (`false` stops the walk). Dummy blocks are skipped. Returns whether the
/// walk reached the end.
pub fn for_each_luma_block(
    rgb: &[u8],
    g: &Geometry,
    qt: &[u16; 64],
    mut f: impl FnMut(usize, usize, &[i16; 64], &[i16; 64]) -> bool,
) -> bool {
    let rc = recips(qt);
    for br in 0..g.y_hib {
        for bc in 0..g.y_wib {
            let mut raw = luma_block(rgb, g, br, bc);
            fdct_islow(&mut raw);
            let mut q = [0i16; 64];
            for i in 0..64 {
                q[i] = rc[i].quantize(raw[i]);
            }
            if !f(br, bc, &raw, &q) {
                return false;
            }
        }
    }
    true
}

/// Encode `rgb` (row-major, 3 bytes per pixel) as a baseline 4:2:0 JPEG at
/// IJG `quality`, byte-identical to libjpeg-turbo `cjpeg -quality Q`.
/// `on_luma` receives every real luma block's pre-quantization islow output
/// and quantized coefficients (for exact side information).
pub fn encode_rgb(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    mut on_luma: impl FnMut(usize, usize, &[i16; 64], &[i16; 64]),
) -> Result<Vec<u8>> {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || w > 65535 || h > 65535 || rgb.len() < w * h * 3 {
        return Err(super::error::JpegError::InvalidDimensions);
    }
    let g = Geometry::new(w, h);
    let qy = scaled_quant_table(&STD_LUMA_QT, quality);
    let qc = scaled_quant_table(&STD_CHROMA_QT, quality);

    // Luma, with jccoefct's dummy blocks (AC 0, DC copied).
    let mut gy = DctGrid::new(g.y_blocks_wide(), g.y_blocks_tall());
    for_each_luma_block(rgb, &g, &qy, |br, bc, raw, q| {
        on_luma(br, bc, raw, q);
        gy.block_mut(br, bc).copy_from_slice(q);
        true
    });
    for mr in 0..g.mcus_tall {
        for mc in 0..g.mcus_wide {
            for yi in 0..2 {
                let br = mr * 2 + yi;
                for xi in 0..2 {
                    let bc = mc * 2 + xi;
                    if g.y_block_is_real(br, bc) {
                        continue;
                    }
                    // Right-edge dummy: DC of the block to its left in the
                    // MCU; bottom dummy row: DC of the MCU's previous block.
                    let dc = if br < g.y_hib {
                        gy.get(br, bc - 1, 0, 0)
                    } else {
                        gy.get(br - 1, mc * 2 + 1, 0, 0)
                    };
                    gy.set(br, bc, 0, 0, dc);
                }
            }
        }
    }

    // Chroma (4:2:0 grids are exactly mcus_wide × mcus_tall blocks).
    let rcc = recips(&qc);
    let mut gcb = DctGrid::new(g.mcus_wide, g.mcus_tall);
    let mut gcr = DctGrid::new(g.mcus_wide, g.mcus_tall);
    for (grid, cb) in [(&mut gcb, true), (&mut gcr, false)] {
        for br in 0..g.mcus_tall {
            for bc in 0..g.mcus_wide {
                let mut d = chroma_block(rgb, &g, br, bc, cb);
                fdct_islow(&mut d);
                let blk = grid.block_mut(br, bc);
                for i in 0..64 {
                    blk[i] = rcc[i].quantize(d[i]);
                }
            }
        }
    }

    // Markers, in libjpeg's order.
    let mut out = Vec::with_capacity(w * h / 4 + 1024);
    out.extend_from_slice(&[0xFF, 0xD8]);
    let seg = |out: &mut Vec<u8>, marker: u8, data: &[u8]| {
        out.extend_from_slice(&[0xFF, marker]);
        out.extend_from_slice(&((data.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(data);
    };
    seg(&mut out, 0xE0, b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00");
    for (id, qt) in [(0u8, &qy), (1u8, &qc)] {
        let mut d = vec![id];
        let mut zz = [0u8; 64];
        for (ni, &v) in qt.iter().enumerate() {
            zz[NATURAL_TO_ZIGZAG[ni]] = v as u8;
        }
        d.extend_from_slice(&zz);
        seg(&mut out, 0xDB, &d);
    }
    let mut sof = vec![8u8];
    sof.extend_from_slice(&(h as u16).to_be_bytes());
    sof.extend_from_slice(&(w as u16).to_be_bytes());
    sof.extend_from_slice(&[3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    seg(&mut out, 0xC0, &sof);
    let specs = [
        HuffmanSpec { class: 0, id: 0, bits: DC_LUMA_BITS, huffval: DC_VALS.to_vec() },
        HuffmanSpec { class: 1, id: 0, bits: AC_LUMA_BITS, huffval: AC_LUMA_VALS.to_vec() },
        HuffmanSpec { class: 0, id: 1, bits: DC_CHROMA_BITS, huffval: DC_VALS.to_vec() },
        HuffmanSpec { class: 1, id: 1, bits: AC_CHROMA_BITS, huffval: AC_CHROMA_VALS.to_vec() },
    ];
    for s in &specs {
        let mut d = vec![(s.class << 4) | s.id];
        d.extend_from_slice(&s.bits);
        d.extend_from_slice(&s.huffval);
        seg(&mut out, 0xC4, &d);
    }
    seg(&mut out, 0xDA, &[3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0]);

    let frame = parse_sof(&sof)?;
    let [dc0, ac0, dc1, ac1] = specs;
    let dc_specs = [Some(dc0), Some(dc1), None, None];
    let ac_specs = [Some(ac0), Some(ac1), None, None];
    let scan_components = [
        ScanComponent { comp_idx: 0, dc_table: 0, ac_table: 0 },
        ScanComponent { comp_idx: 1, dc_table: 1, ac_table: 1 },
        ScanComponent { comp_idx: 2, dc_table: 1, ac_table: 1 },
    ];
    let scan = encode_scan(&frame, &scan_components, &[gy, gcb, gcr], &dc_specs, &ac_specs, 0)?;
    out.extend_from_slice(&scan);
    out.extend_from_slice(&[0xFF, 0xD9]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_quantizer_matches_rounding_division() {
        // libjpeg-turbo's reciprocal quantizer equals round-half-up
        // division of |v| for the islow range, for every baseline divisor.
        for q in 1u32..=255 {
            let d = q << 3;
            let rc = Recip::new(d);
            for v in -16384i32..16384 {
                let a = v.unsigned_abs();
                let want = ((a + d / 2) / d) as i32 * v.signum();
                assert_eq!(rc.quantize(v as i16) as i32, want, "q={q} v={v}");
            }
        }
    }

    #[test]
    fn quant_tables_scale_like_ijg() {
        assert_eq!(scaled_quant_table(&STD_LUMA_QT, 50), STD_LUMA_QT);
        let q75 = scaled_quant_table(&STD_LUMA_QT, 75);
        assert_eq!(&q75[..4], &[8, 6, 5, 8]);
        assert_eq!(scaled_quant_table(&STD_LUMA_QT, 100), [1u16; 64]);
        assert_eq!(scaled_quant_table(&STD_LUMA_QT, 1)[0], 255);
    }

    /// Cross-platform pin (native aarch64 / x86_64 and WASM must agree; the
    /// value equals libjpeg-turbo 3.1 `cjpeg -quality 75` on this image).
    #[test]
    fn output_is_pinned() {
        use sha2::{Digest, Sha256};
        let (w, h) = (77usize, 45usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i * 37 + (i / 7) * 11) % 251) as u8).collect();
        let bytes = encode_rgb(&rgb, w as u32, h as u32, 75, |_, _, _, _| {}).unwrap();
        let hex: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "543d527e4a8e01bff6271b54cf5173c598a7e6a50ea427483fc0952dc2a06712");
    }

    #[test]
    fn output_parses_and_matches_luma() {
        let (w, h) = (37usize, 21usize);
        let mut s = 0x2545F491u32;
        let rgb: Vec<u8> = (0..w * h * 3)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 24) as u8
            })
            .collect();
        let mut blocks = Vec::new();
        let bytes = encode_rgb(&rgb, w as u32, h as u32, 85, |br, bc, _, q| blocks.push((br, bc, *q))).unwrap();
        let img = super::super::JpegImage::from_bytes(&bytes).unwrap();
        assert_eq!(img.frame_info().width as usize, w);
        assert_eq!(blocks.len(), w.div_ceil(8) * h.div_ceil(8));
        for (br, bc, q) in blocks {
            assert_eq!(img.dct_grid(0).block(br, bc), &q[..]);
        }
        // Re-serializing the parsed image is lossless.
        assert_eq!(img.to_bytes().unwrap(), bytes);
    }
}
