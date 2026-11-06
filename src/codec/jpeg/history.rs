// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Does a pixel buffer come from decoding a JPEG on the 8×8 grid a cover
//! written from it would use? (JPEG compression-history test.)
//!
//! Deep Cover's exact side information assumes the pixels are a *precover*:
//! an image whose DCT coefficients, before the cover's quantization, are
//! continuous, so the rounding errors are spread evenly over `[−½, ½]`. A
//! decoded JPEG re-written at the same size (no resample, or a transform that
//! keeps the block grid, like a 90° rotation of a frame whose sides are
//! multiples of 8) is not one: every block's DCT sits on the source's
//! quantization lattice `q₁·c` (up to the decoder's pixel rounding), the
//! re-quantization at `q₂` leaves a comb in the coefficient histogram, and
//! the rounding errors cluster at 0 and ±½. Exact SI then makes the
//! near-½ coefficients nearly free and they fill the comb's empty bins: a
//! histogram attack reached P_E 0.04 at 0.20 bpnzAC
//! (`eval/runs/2026-10-31-wand-ghost/RESULTS.md`, arm D). The cover
//! writer's exactness check cannot see this — the cover *is* the writer's
//! output for those pixels.
//!
//! A 90° rotation of a decoded JPEG whose height is **not** a multiple of 8
//! moves the source's grid by `h mod 8` along the new x axis: the cover's
//! grid is no longer the source's on that axis, but it still is on the
//! other, the lattice shows there, and the test still flags such pixels
//! (R3 review 2026-11-02; test `quarter_turn_off_grid_is_still_flagged`).
//! That errs on the safe side: they get no side information and nzAC / 10
//! although the cover is not an aligned double compression.
//!
//! The test is the classic one for JPEG history of a bitmap (Fan & de
//! Queiroz 2003 quantizer estimation; Luo, Huang & Qiu 2010 "JPEG error
//! analysis"): take the 8×8 DCT of the luma on the grid and look, per AC
//! frequency, for a lattice of integer step `q`. For each frequency `k` and
//! step `q`, over the coefficients outside the zero cell (`|D| ≥ q/2`, the
//! cell every smooth block falls into), the lattice phase
//! `t = 1 − |D − q·round(D/q)| / (q/4)` has mean 0 and variance 1/3 for a
//! precover (first-order density slopes cancel between the two signs) and
//! mean ≈ 1 − 4σ/q for a decoded JPEG with pixel-rounding noise σ ≈ 0.3.
//! `z = Σt / √(n/3)` is then a standard score:
//!
//! - `lattice_z` — per AC frequency the best `z` over steps `q ≥ 2` (any JPEG
//!   at quality ≤ 98 has `q ≥ 2` somewhere), and of those the **third
//!   largest** (or the second largest, at a higher bar, for smooth images at
//!   low quality that populate only two frequencies): a decoded JPEG shows
//!   its lattice on many frequencies at once,
//!   while a precover can show one spurious "lattice" on one or two (a smooth
//!   gradient puts most blocks' (1,0) coefficient near one value, which then
//!   sits near a multiple of some `q`: BOSSbase 5386, peak `z` 24);
//! - `unit_z` — the step-1 scores of all frequencies pooled (quality 99–100,
//!   where every step is 1 and only the integer lattice is left).
//!
//! Integer arithmetic up to the final square root (IEEE-exact), so the
//! answer is the same on every architecture and on WASM. Cost: at most
//! [`MAX_BLOCKS`] blocks, a few ms.

use super::cover_writer::{luma_block, Geometry};
use super::pixels_aan::fdct_islow;
use super::zigzag::NATURAL_TO_ZIGZAG;
use std::collections::HashSet;

/// Blocks examined at most (evenly spaced over the image).
pub const MAX_BLOCKS: usize = 16_384;

/// Largest step tested on the 14 lowest AC frequencies (IJG quality ≈ 10
/// still has steps ≤ 64 there) and on the rest.
const QMAX_LOW: usize = 64;
const QMAX_HIGH: usize = 24;
const LOW_ZIGZAG: usize = 15;

/// Fewest coefficients a (frequency, step) test needs to count.
const NMIN: u32 = 256;

/// Decision thresholds (standard scores). Measured on BOSSbase raw, PNG,
/// HEIC-derived and resampled-JPEG pixels (precovers) against same-size
/// decoded JPEGs; `docs/design/image/deep-cover-exact-si.md`, "Decoded-JPEG
/// guard".
/// `lattice_z` (third-largest frequency) threshold.
pub const LATTICE_Z_MIN: f64 = 10.0;
/// `lattice_z2` (second-largest frequency) threshold: a smooth decoded JPEG
/// at low quality may populate only two frequencies, but strongly.
pub const LATTICE2_Z_MIN: f64 = 15.0;
/// `unit_z` threshold.
pub const UNIT_Z_MIN: f64 = 10.0;

/// What the lattice test found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridEvidence {
    /// Distinct non-flat full blocks examined.
    pub blocks: usize,
    /// Third- and second-largest per-frequency lattice scores (steps ≥ 2):
    /// the decision statistics.
    pub lattice_z: f64,
    pub lattice_z2: f64,
    /// Largest lattice score over AC frequencies and steps ≥ 2, with its
    /// natural index and step.
    pub peak_z: f64,
    pub peak_k: usize,
    pub peak_q: usize,
    /// Pooled step-1 score.
    pub unit_z: f64,
    /// Best score over steps ≥ 2 per AC frequency (natural order; 0 = DC / untested).
    pub freq_z: [f64; 64],
}

impl GridEvidence {
    /// The pixels are (almost certainly) a JPEG decoded on this grid.
    pub fn is_decoded_jpeg(&self) -> bool {
        self.lattice_z >= LATTICE_Z_MIN || self.lattice_z2 >= LATTICE2_Z_MIN || self.unit_z >= UNIT_Z_MIN
    }
}

fn qmax(k: usize) -> usize {
    if (NATURAL_TO_ZIGZAG[k] as usize) < LOW_ZIGZAG { QMAX_LOW } else { QMAX_HIGH }
}

fn block_hash(b: &[i16; 64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &v in b {
        h ^= v as u16 as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Lattice evidence of `rgb` (row-major, 3 bytes per pixel, `width × height`)
/// on the 8×8 grid anchored at the top-left corner — the grid of the Deep
/// Cover writer's cover for these pixels. Only full blocks count (a partial
/// edge block of the source was padded with pixels the decode does not
/// have). Identical blocks count once (screenshots and graphics repeat
/// blocks, which would otherwise look like a lattice).
pub fn grid_evidence(rgb: &[u8], width: u32, height: u32) -> GridEvidence {
    let (w, h) = (width as usize, height as usize);
    let (fbw, fbh) = (w / 8, h / 8);
    let nb = fbw * fbh;
    let mut ev = GridEvidence { blocks: 0, lattice_z: 0.0, lattice_z2: 0.0, peak_z: 0.0, peak_k: 0, peak_q: 0, unit_z: 0.0, freq_z: [0.0; 64] };
    if nb == 0 || rgb.len() < w * h * 3 {
        return ev;
    }
    let g = Geometry::new(w, h);
    let take = nb.min(MAX_BLOCKS);
    let mut seen: HashSet<u64> = HashSet::with_capacity(take);
    let mut acc = vec![0i64; 64 * (QMAX_LOW + 1)];
    let mut cnt = vec![0u32; 64 * (QMAX_LOW + 1)];
    let qmaxes: [usize; 64] = std::array::from_fn(qmax);
    for j in 0..take {
        let idx = (j as u64 * nb as u64 / take as u64) as usize;
        let (br, bc) = (idx / fbw, idx % fbw);
        let mut b = luma_block(rgb, &g, br, bc);
        if !seen.insert(block_hash(&b)) {
            continue;
        }
        fdct_islow(&mut b);
        // islow output is 8 × the orthonormal DCT: D = raw / 8.
        let mut any = false;
        for k in 1..64 {
            let v = b[k] as i32;
            let a = v.abs();
            if a < 4 {
                continue; // |D| < ½: in the zero cell for every step
            }
            any = true;
            for q in 1..=qmaxes[k] {
                let q = q as i32;
                if a < 4 * q {
                    break; // zero cell of this and every larger step
                }
                let m = 8 * q;
                let mut r = v.rem_euclid(m);
                if r > 4 * q {
                    r -= m;
                }
                let i = k * (QMAX_LOW + 1) + q as usize;
                acc[i] += (2 * q - r.abs()) as i64; // 2q · t
                cnt[i] += 1;
            }
        }
        if any {
            ev.blocks += 1;
        }
    }
    let z = |i: usize, q: usize| acc[i] as f64 * 3f64.sqrt() / (2.0 * q as f64 * (cnt[i] as f64).sqrt());
    let (mut unit_sum, mut unit_n) = (0.0f64, 0usize);
    for k in 1..64 {
        for q in 1..=qmaxes[k] {
            let i = k * (QMAX_LOW + 1) + q;
            if cnt[i] < NMIN {
                continue;
            }
            let zi = z(i, q);
            if q == 1 {
                unit_sum += zi;
                unit_n += 1;
            } else if zi > ev.freq_z[k] {
                ev.freq_z[k] = zi;
            }
            if q > 1 && zi > ev.peak_z {
                ev.peak_z = zi;
                ev.peak_k = k;
                ev.peak_q = q;
            }
        }
    }
    let mut best = ev.freq_z;
    best.sort_unstable_by(|a, b| b.total_cmp(a));
    ev.lattice_z = best[2];
    ev.lattice_z2 = best[1];
    if unit_n > 0 {
        ev.unit_z = unit_sum / (unit_n as f64).sqrt();
    }
    ev
}

/// [`grid_evidence`]`(..).is_decoded_jpeg()`.
pub fn is_aligned_jpeg_decode(rgb: &[u8], width: u32, height: u32) -> bool {
    grid_evidence(rgb, width, height).is_decoded_jpeg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::jpeg::cover_writer::encode_rgb;

    /// A smooth gradient with texture and sensor-like noise (a precover).
    fn photo(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        let mut noise = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 24) as f64 / 255.0 - 0.5
        };
        let mut v = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f64, y as f64);
                let t = 120.0 + 60.0 * crate::det_math::det_sin(fx * 0.071 + fy * 0.023) * crate::det_math::det_cos(fy * 0.05);
                for c in 0..3 {
                    let val = t + 20.0 * c as f64 - 20.0 + 12.0 * noise();
                    v.push(val.round().clamp(0.0, 255.0) as u8);
                }
            }
        }
        v
    }

    fn decode(jpeg: &[u8]) -> Vec<u8> {
        image::load_from_memory(jpeg).unwrap().to_rgb8().into_raw()
    }

    fn rot90(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
        // (x, y) → (h−1−y, x) in an h × w image.
        let mut o = vec![0u8; rgb.len()];
        for y in 0..h {
            for x in 0..w {
                let (nx, ny) = (h - 1 - y, x);
                o[(ny * h + nx) * 3..(ny * h + nx) * 3 + 3].copy_from_slice(&rgb[(y * w + x) * 3..(y * w + x) * 3 + 3]);
            }
        }
        o
    }

    #[test]
    fn precovers_are_not_decoded_jpegs() {
        for (w, h, seed) in [(320usize, 240usize, 1u32), (257, 199, 2), (512, 512, 3)] {
            let e = grid_evidence(&photo(w, h, seed), w as u32, h as u32);
            assert!(!e.is_decoded_jpeg(), "{w}x{h}: {e:?}");
        }
        // Flat or tiny images carry no evidence either way.
        assert!(!is_aligned_jpeg_decode(&vec![128u8; 64 * 64 * 3], 64, 64));
        assert!(!is_aligned_jpeg_decode(&[1, 2, 3], 1, 1));
    }

    #[test]
    fn same_size_decoded_jpegs_are_detected_at_every_quality() {
        let (w, h) = (320usize, 240usize);
        let pre = photo(w, h, 7);
        for q in [30u8, 50, 75, 85, 92, 95, 98, 100] {
            let px = decode(&encode_rgb(&pre, w as u32, h as u32, q, |_, _, _, _| {}).unwrap());
            let e = grid_evidence(&px, w as u32, h as u32);
            assert!(e.is_decoded_jpeg(), "q{q}: {e:?}");
            // A quarter-turn of the decode keeps the grid (sides multiples of 8).
            assert!(is_aligned_jpeg_decode(&rot90(&px, w, h), h as u32, w as u32), "q{q} rotated");
            // A 4-pixel crop moves the grid: no longer aligned. (Not at q30:
            // there most blocks are flat and the 4-px-shifted block edges
            // between them are quantized DC steps, a lattice of their own;
            // the guard then errs on the safe side.)
            if q >= 50 {
                let crop: Vec<u8> = (4..h).flat_map(|y| px[(y * w + 4) * 3..(y + 1) * w * 3].to_vec()).collect();
                let e = grid_evidence(&crop, (w - 4) as u32, (h - 4) as u32);
                assert!(!e.is_decoded_jpeg(), "q{q} cropped: {e:?}");
            }
        }
    }

    /// A quarter-turn of a decode whose height is not a multiple of 8 keeps
    /// the grid along one axis only; still flagged (the safe side, module
    /// docs).
    #[test]
    fn quarter_turn_off_grid_is_still_flagged() {
        let (w, h) = (320usize, 236usize); // h mod 8 = 4
        let pre = photo(w, h, 9);
        for q in [75u8, 85, 92] {
            let px = decode(&encode_rgb(&pre, w as u32, h as u32, q, |_, _, _, _| {}).unwrap());
            let e = grid_evidence(&rot90(&px, w, h), h as u32, w as u32);
            assert!(e.is_decoded_jpeg(), "q{q}: {e:?}");
        }
    }

    #[test]
    fn identical_blocks_count_once() {
        // A screenshot-like image: one textured block repeated everywhere
        // would otherwise score as a perfect lattice.
        let tile = photo(8, 8, 11);
        let (w, h) = (256usize, 256usize);
        let rgb: Vec<u8> = (0..w * h).flat_map(|i| {
            let (x, y) = (i % w % 8, i / w % 8);
            tile[(y * 8 + x) * 3..(y * 8 + x) * 3 + 3].to_vec()
        }).collect();
        let e = grid_evidence(&rgb, w as u32, h as u32);
        assert_eq!(e.blocks, 1);
        assert!(!e.is_decoded_jpeg());
    }
}
