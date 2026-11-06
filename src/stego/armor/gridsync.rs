// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Key-free geometric synchronisation for Fortress T: a block-checkerboard
//! sync pattern, found in the received luma's spectrum.
//!
//! # The pattern
//!
//! The Fortress T encoder adds `a_b * c(bx, by)` to block averages before their
//! QIM, with `c = +1` where `bx + by` is even and `-1` elsewhere (a checkerboard
//! of 8 x 8 blocks) and `a_b` large in busy blocks, zero in flat ones (see
//! `fortress_tile`). QIM keeps the offset in expectation (its rounding error
//! does not depend on it), so the pattern is in the image, and costs nothing at
//! the QIM decoder. In pixels `c(p) = sq(px / 16) sq(py / 16)`, sq a square wave
//! of period 16 (+1 on [0, 8)), whose strongest spectral lines are
//! `k1 = (1, -1) / 16` and `k2 = (1, 1) / 16` cycles/px:
//! `c(p) ~ (8 / pi^2) [cos(2 pi k1.p) - cos(2 pi k2.p)]`.
//!
//! This is a periodic sync template in the sense of Kalker et al. (1999) and
//! Kutter (1998), reduced to its strongest form: one pair of lines, as high in
//! frequency as a block-level pattern goes (0.088 cycles/px), on the diagonals,
//! away from the axis-aligned harmonics any JPEG grid prints.
//!
//! # The estimator
//!
//! 1. Coarse: magnitude spectrum of the (box-decimated, so the long side is at
//!    most 1024) luma, Hann-windowed and zero-padded to N x N, normalised by
//!    its local background; search for the line pair over rotation +-10 deg
//!    and scale 0.4-1.7; local maxima above a gate are candidates.
//! 2. Fine: sub-bin peak (log-parabola), then the phase gradient between the
//!    halves of the full-resolution image (direct sums, two iterations).
//! 3. Phase: the whole-image sums' phases place the pattern, so the encoder's
//!    block corners (up to whole blocks; the keyed pilots resolve those).
//!
//! The result is an affine map from the encoder's pixel frame to the received
//! image (rotation, scale, also slight anisotropy and shear, and translation).
//!
//! Deterministic: IEEE basic operations, `sqrt`, `det_sincos`, `det_atan2`,
//! `det_ln` and the in-house FFT; parallel sums are reduced in a fixed order.

use crate::det_math::{det_atan2, det_ln, det_sincos};
use crate::stego::armor::fft2d;

use std::f64::consts::{PI, SQRT_2, TAU};

/// Largest rotation searched (degrees, either sign).
const MAX_ROT_DEG: f64 = 10.0;
/// Scale range searched. Up to 2.6 since 2026-11-02: Instagram enlarges
/// narrow (tall) covers 1.9-2.4x, which 1.7 missed (the search locked onto a
/// harmonic; eval/runs/2026-11-02-armor-geometry-2).
const MIN_SCALE: f64 = 0.4;
const MAX_SCALE: f64 = 2.6;
/// Candidates tried at most (best first).
pub const CANDIDATES: usize = 3;
/// Gate on the coarse pair score (two background-normalised magnitudes). A
/// line-free spectrum's best local maxima score about 4.5-5.8.
pub const SCORE_MIN: f64 = 6.0;
/// Radius of the checkerboard lines at scale 1 (cycles/px).
const BASE: f64 = SQRT_2 / 16.0;

/// A pixel value of a [`Luma`] plane: f32 (the decoder's planes, 4 B per
/// pixel) or f64.
pub trait Px: Copy + Into<f64> + Send + Sync {}
impl Px for f32 {}
impl Px for f64 {}

/// A luma plane (row stride `stride`, real area `w` x `h`).
#[derive(Clone, Copy)]
pub struct Luma<'a, T = f64> {
    pub px: &'a [T],
    pub stride: usize,
    pub w: usize,
    pub h: usize,
}

impl<T: Px> Luma<'_, T> {
    #[inline]
    fn at(&self, x: usize, y: usize) -> f64 {
        self.px[y * self.stride + x].into()
    }

    /// Bilinear sample at continuous pixel-index coordinates (edge-clamped).
    #[inline]
    pub fn bilinear(&self, x: f64, y: f64) -> f64 {
        let x = x.clamp(0.0, (self.w - 1) as f64);
        let y = y.clamp(0.0, (self.h - 1) as f64);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (xi, yi) = (x0 as usize, y0 as usize);
        let xj = (xi + 1).min(self.w - 1);
        let yj = (yi + 1).min(self.h - 1);
        let a = self.at(xi, yi) * (1.0 - fx) + self.at(xj, yi) * fx;
        let b = self.at(xi, yj) * (1.0 - fx) + self.at(xj, yj) * fx;
        a * (1.0 - fy) + b * fy
    }
}

/// Affine map from the encoder's pixel frame to the received image:
/// `X = o + A p` in continuous coordinates (pixel `u` spans `[u, u + 1)`);
/// `p = (0, 0)` is *some* corner of the encoder's block grid.
#[derive(Clone, Copy, Debug)]
pub struct GridMap {
    pub a: [[f64; 2]; 2],
    pub o: [f64; 2],
    /// Coarse score of the candidate it came from.
    pub score: f64,
}

impl GridMap {
    pub fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        [
            self.o[0] + self.a[0][0] * p[0] + self.a[0][1] * p[1],
            self.o[1] + self.a[1][0] * p[0] + self.a[1][1] * p[1],
        ]
    }

    /// True if this is the received block grid itself (no rotation, no
    /// rescale, block corners on the received ones).
    pub fn is_identity(&self) -> bool {
        let near = |v: f64| {
            let m = v.rem_euclid(8.0);
            m.min(8.0 - m) < 0.25
        };
        (self.a[0][0] - 1.0).abs() < 5e-4
            && (self.a[1][1] - 1.0).abs() < 5e-4
            && self.a[0][1].abs() < 5e-4
            && self.a[1][0].abs() < 5e-4
            && near(self.o[0])
            && near(self.o[1])
    }

    /// Rotation (degrees) and mean scale, for reporting.
    pub fn rotation_scale(&self) -> (f64, f64) {
        let rot = det_atan2(self.a[1][0], self.a[0][0]).to_degrees();
        let det = self.a[0][0] * self.a[1][1] - self.a[0][1] * self.a[1][0];
        (rot, det.abs().sqrt())
    }
}

/// A coarse checkerboard candidate: the two line frequencies (cycles/px of the
/// received image) and the coarse score.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    k1: [f64; 2],
    k2: [f64; 2],
    pub score: f64,
}

/// Background-normalised magnitude spectrum of the decimated, windowed luma.
struct Spec {
    n: usize,
    /// Box decimation factor of the luma.
    d: usize,
    norm: Vec<f64>,
    raw: Vec<f64>,
}

/// Cyclic (2r+1) x (2r+1) box sums of an n x n plane.
fn box_sum(m: &[f64], n: usize, r: usize) -> Vec<f64> {
    let mut tmp = vec![0.0f64; n * n];
    for v in 0..n {
        let row = &m[v * n..(v + 1) * n];
        let mut s: f64 = (0..=2 * r).map(|i| row[(i + n - r) % n]).sum();
        for u in 0..n {
            tmp[v * n + u] = s;
            s += row[(u + r + 1) % n] - row[(u + n - r) % n];
        }
    }
    let mut out = vec![0.0f64; n * n];
    for u in 0..n {
        let mut s: f64 = (0..=2 * r).map(|i| tmp[((i + n - r) % n) * n + u]).sum();
        for v in 0..n {
            out[v * n + u] = s;
            s += tmp[((v + r + 1) % n) * n + u] - tmp[((v + n - r) % n) * n + u];
        }
    }
    out
}

fn hann(len: usize) -> Vec<f64> {
    (0..len).map(|i| 0.5 - 0.5 * det_sincos(TAU * (i as f64 + 0.5) / len as f64).1).collect()
}

/// Plain (edge-truncated) box mean of radius r of a w x h plane.
fn box_mean(m: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let mut ii = vec![0.0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut s = 0.0;
        for x in 0..w {
            s += m[y * w + x];
            ii[(y + 1) * (w + 1) + x + 1] = ii[y * (w + 1) + x + 1] + s;
        }
    }
    let mut out = vec![0.0f64; w * h];
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            let s = ii[y1 * (w + 1) + x1] - ii[y0 * (w + 1) + x1] - ii[y1 * (w + 1) + x0] + ii[y0 * (w + 1) + x0];
            out[y * w + x] = s / ((y1 - y0) * (x1 - x0)) as f64;
        }
    }
    out
}

/// Floor of the local variance in [`whiten_local`] (gray levels^2).
const WHITEN_EPS: f64 = 4.0;

/// Local whitening before the search and the line refinement: high-pass (a
/// box about one pattern period wide), divided by the local standard
/// deviation (a box of 4 blocks), so busy areas (strong content at the line
/// frequencies) do not drown the pattern carried by quieter ones. Worth about
/// 1.4x sync amplitude at an unchanged false-peak level; it let the encoder
/// drop the amplitude from 2.0 to 1.25 (eval/runs/2026-11-02-armor-geometry-2,
/// section 2). `d` is the decimation of the plane. Box sums and `sqrt` only.
fn whiten_local(data: &mut [f64], w: usize, h: usize, d: usize) {
    let r1 = (8 / d).max(1) / 2 * 2; // box ~ one pattern period wide
    let low = box_mean(data, w, h, r1.max(1));
    let hp: Vec<f64> = data.iter().zip(&low).map(|(a, b)| a - b).collect();
    let sq: Vec<f64> = hp.iter().map(|v| v * v).collect();
    let var = box_mean(&sq, w, h, (32 / d).max(4));
    for i in 0..data.len() {
        data[i] = hp[i] / (var[i] + WHITEN_EPS).sqrt();
    }
}

/// The full-resolution luma, locally whitened as [`whiten_local`] (d = 1),
/// for the line refinement; f32, `w` x `h` rows.
///
/// Streamed by rows: the high-pass from running column sums of the luma
/// rows, the local variance from running column sums of the squares of a ring
/// of `2 R2 + 2` high-passed rows, so beyond the result (4 B per pixel) it
/// keeps a few rows. Whole-image planes (high-pass, squares, box means,
/// integral image) took about 48 B per pixel, 3 GB more at 48 MP.
/// Cancellable (every 256 rows).
pub fn whitened_full<T: Px>(l: &Luma<T>) -> Result<Vec<f32>, crate::stego::error::StegoError> {
    let (w, h) = (l.w, l.h);
    let mut out = vec![0.0f32; w * h];
    if w == 0 || h == 0 {
        return Ok(out);
    }
    let (r1, r2) = (8 / 2 * 2, 32);
    let row = |y: usize| &l.px[y * l.stride..y * l.stride + w];
    // Horizontal box sums (edge-truncated) of `col` into `dst`, divided by the
    // window's cell count (`rows` rows times its clipped width).
    let hmean = |col: &[f64], r: usize, rows: usize, pre: &mut [f64], dst: &mut [f64]| {
        for x in 0..w {
            pre[x + 1] = pre[x] + col[x];
        }
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            dst[x] = (pre[x1] - pre[x0]) / ((x1 - x0) * rows) as f64;
        }
    };
    let ring_n = 2 * r2 + 2;
    let mut ring = vec![0.0f64; ring_n * w];
    let (mut col1, mut col2) = (vec![0.0f64; w], vec![0.0f64; w]);
    let (mut pre, mut tmp) = (vec![0.0f64; w + 1], vec![0.0f64; w]);
    let (mut lo1, mut hi1, mut lo2, mut hi2, mut next) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for y in 0..h {
        if y % 256 == 0 {
            crate::stego::progress::check_cancelled()?;
        }
        // High-passed rows up to y + R2.
        while next < h && next <= y + r2 {
            let (a, b) = (next.saturating_sub(r1), (next + r1 + 1).min(h));
            while hi1 < b {
                col1.iter_mut().zip(row(hi1)).for_each(|(c, &v)| *c += v.into());
                hi1 += 1;
            }
            while lo1 < a {
                col1.iter_mut().zip(row(lo1)).for_each(|(c, &v)| *c -= v.into());
                lo1 += 1;
            }
            hmean(&col1, r1, b - a, &mut pre, &mut tmp);
            let slot = &mut ring[(next % ring_n) * w..(next % ring_n + 1) * w];
            for ((o, &v), &m) in slot.iter_mut().zip(row(next)).zip(&tmp) {
                *o = v.into() - m;
            }
            next += 1;
        }
        let (a, b) = (y.saturating_sub(r2), (y + r2 + 1).min(h));
        while hi2 < b {
            let r = &ring[(hi2 % ring_n) * w..(hi2 % ring_n + 1) * w];
            col2.iter_mut().zip(r).for_each(|(c, &v)| *c += v * v);
            hi2 += 1;
        }
        while lo2 < a {
            let r = &ring[(lo2 % ring_n) * w..(lo2 % ring_n + 1) * w];
            col2.iter_mut().zip(r).for_each(|(c, &v)| *c -= v * v);
            lo2 += 1;
        }
        hmean(&col2, r2, b - a, &mut pre, &mut tmp);
        let hp = &ring[(y % ring_n) * w..(y % ring_n + 1) * w];
        for ((o, &v), &var) in out[y * w..(y + 1) * w].iter_mut().zip(hp).zip(&tmp) {
            *o = (v / (var.max(0.0) + WHITEN_EPS).sqrt()) as f32;
        }
    }
    Ok(out)
}

impl Spec {
    fn build<T: Px>(l: &Luma<T>) -> Option<Spec> {
        let d = l.w.max(l.h).div_ceil(1024).clamp(1, 4);
        let (wd, hd) = (l.w / d, l.h / d);
        if wd.min(hd) < 128 {
            return None;
        }
        let n = if wd.max(hd) > 512 { 1024 } else if wd.max(hd) > 256 { 512 } else { 256 };
        let (dw, dh) = (wd.min(n), hd.min(n));
        let (xo, yo) = ((wd - dw) / 2, (hd - dh) / 2);
        // Box-decimated data region.
        let mut data = vec![0.0f64; dw * dh];
        let inv = 1.0 / (d * d) as f64;
        for y in 0..dh {
            for x in 0..dw {
                let (sx, sy) = ((xo + x) * d, (yo + y) * d);
                let mut s = 0.0;
                for j in 0..d {
                    for i in 0..d {
                        s += l.at(sx + i, sy + j);
                    }
                }
                data[y * dw + x] = s * inv;
            }
        }
        whiten_local(&mut data, dw, dh, d);
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let (hx, hy) = (hann(dw), hann(dh));
        let mut win = vec![0.0f64; n * n];
        for y in 0..dh {
            for x in 0..dw {
                win[y * n + x] = (data[y * dw + x] - mean) * hx[x] * hy[y];
            }
        }
        let spec = fft2d::fft2d(&win, n, n);
        let raw: Vec<f64> = spec
            .data
            .iter()
            .map(|c| ((c.re as f64) * (c.re as f64) + (c.im as f64) * (c.im as f64)).sqrt())
            .collect();
        // Local background: the (2R+1)^2 box without the central (2E+1)^2,
        // E covering the Hann main lobe, which zero padding widens by n / dw.
        let e = (2 * n).div_ceil(dw.min(dh)).max(1);
        let r = e + 6;
        let big = box_sum(&raw, n, r);
        let small = box_sum(&raw, n, e);
        let cnt = ((2 * r + 1) * (2 * r + 1) - (2 * e + 1) * (2 * e + 1)) as f64;
        let norm = raw.iter().enumerate().map(|(i, &m)| m / ((big[i] - small[i]) / cnt).max(1e-9)).collect();
        Some(Spec { n, d, norm, raw })
    }

    /// Bilinear normalised magnitude at fractional (signed) bin coordinates.
    fn at(&self, u: f64, v: f64) -> f64 {
        let n = self.n as isize;
        let (u0, v0) = (u.floor(), v.floor());
        let (fu, fv) = (u - u0, v - v0);
        let g = |a: isize, b: isize| self.norm[(b.rem_euclid(n) * n + a.rem_euclid(n)) as usize];
        let (a, b) = (u0 as isize, v0 as isize);
        (g(a, b) * (1.0 - fu) + g(a + 1, b) * fu) * (1.0 - fv) + (g(a, b + 1) * (1.0 - fu) + g(a + 1, b + 1) * fu) * fv
    }

    /// Pair score: lines at radius `r` bins, rotation `t`, angles t -+ 45 deg.
    fn pair(&self, r: f64, t: f64) -> f64 {
        let (s1, c1) = det_sincos(t - PI / 4.0);
        let (s2, c2) = det_sincos(t + PI / 4.0);
        self.at(r * c1, r * s1) + self.at(r * c2, r * s2)
    }

    /// Bins per cycle/px of the received image.
    fn bins_per_cycle(&self) -> f64 {
        (self.n * self.d) as f64
    }

    /// Sub-bin location (received cycles/px) of the line near bin (u, v): the
    /// largest raw magnitude within +-2 bins, then a separable log-parabola.
    fn peak(&self, u: f64, v: f64) -> [f64; 2] {
        let n = self.n as isize;
        let g = |a: isize, b: isize| self.raw[(b.rem_euclid(n) * n + a.rem_euclid(n)) as usize].max(1e-12);
        let (u0, v0) = (u.round() as isize, v.round() as isize);
        let (mut bu, mut bv, mut bm) = (u0, v0, f64::MIN);
        for dv in -2..=2 {
            for du in -2..=2 {
                let m = g(u0 + du, v0 + dv);
                if m > bm {
                    (bu, bv, bm) = (u0 + du, v0 + dv, m);
                }
            }
        }
        let para = |a: f64, b: f64, c: f64| {
            let (la, lb, lc) = (det_ln(a), det_ln(b), det_ln(c));
            let den = la - 2.0 * lb + lc;
            if den.abs() < 1e-12 { 0.0 } else { (0.5 * (la - lc) / den).clamp(-0.5, 0.5) }
        };
        let du = para(g(bu - 1, bv), bm, g(bu + 1, bv));
        let dv = para(g(bu, bv - 1), bm, g(bu, bv + 1));
        let bpc = self.bins_per_cycle();
        [(bu as f64 + du) / bpc, (bv as f64 + dv) / bpc]
    }

    /// Up to `k` local maxima (r bins, t rad, score) of the pair score above
    /// the gate, best first, each refined on a finer local grid.
    fn search(&self, k: usize) -> Vec<(f64, f64, f64)> {
        let bpc = self.bins_per_cycle();
        let rmin = BASE / MAX_SCALE * bpc;
        let rmax = (BASE / MIN_SCALE * bpc).min(0.45 * self.n as f64);
        if rmax <= rmin {
            return Vec::new();
        }
        let tmax = MAX_ROT_DEG.to_radians();
        let tstep = 0.25f64.to_radians();
        let rstep = 0.5;
        let nt = (2.0 * tmax / tstep).round() as usize + 1;
        let nr = ((rmax - rmin) / rstep).floor() as usize + 1;
        let score_row = |it: usize| -> Vec<f64> {
            let t = -tmax + it as f64 * tstep;
            (0..nr).map(|ir| self.pair(rmin + ir as f64 * rstep, t)).collect()
        };
        #[cfg(feature = "parallel")]
        let rows: Vec<Vec<f64>> = {
            use rayon::prelude::*;
            (0..nt).into_par_iter().map(score_row).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let rows: Vec<Vec<f64>> = (0..nt).map(score_row).collect();
        let mut peaks: Vec<(f64, f64, f64)> = Vec::new();
        for it in 0..nt {
            for ir in 0..nr {
                let v = rows[it][ir];
                if v < SCORE_MIN {
                    continue;
                }
                let mut is_max = true;
                'nb: for dt in -1i64..=1 {
                    for dr in -1i64..=1 {
                        let (a, b) = (it as i64 + dt, ir as i64 + dr);
                        if (dt, dr) == (0, 0) || a < 0 || b < 0 || a >= nt as i64 || b >= nr as i64 {
                            continue;
                        }
                        if rows[a as usize][b as usize] > v {
                            is_max = false;
                            break 'nb;
                        }
                    }
                }
                if is_max {
                    peaks.push((rmin + ir as f64 * rstep, -tmax + it as f64 * tstep, v));
                }
            }
        }
        peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        peaks.truncate(k);
        peaks
            .into_iter()
            .map(|(mut r0, mut t0, mut s0)| {
                let (rc, tc) = (r0, t0);
                for i in -5i64..=5 {
                    for j in -5i64..=5 {
                        let (r, t) = (rc + i as f64 * 0.1, tc + j as f64 * 0.05f64.to_radians());
                        let s = self.pair(r, t);
                        if s > s0 {
                            (r0, t0, s0) = (r, t, s);
                        }
                    }
                }
                (r0, t0, s0)
            })
            .collect()
    }
}

/// Checkerboard candidates in a received luma plane, best first (at most
/// [`CANDIDATES`], each above [`SCORE_MIN`]). Key-free.
pub fn checker_candidates<T: Px>(l: &Luma<T>) -> Vec<Candidate> {
    let Some(spec) = Spec::build(l) else { return Vec::new() };
    spec.search(CANDIDATES)
        .into_iter()
        .map(|(r, t, score)| {
            let (s1, c1) = det_sincos(t - PI / 4.0);
            let (s2, c2) = det_sincos(t + PI / 4.0);
            Candidate { k1: spec.peak(r * c1, r * s1), k2: spec.peak(r * c2, r * s2), score }
        })
        .collect()
}

/// Sums of the luma at frequency `k` over every `step`-th row: S = sum L e,
/// Sx = sum L (x - cx) e, Sy = sum L (y - cy) e (e = exp(-2 pi i k.X), X the
/// pixel centre, c the image centre), and the coordinate variances.
fn moment_sums<T: Px>(l: &Luma<T>, k: [f64; 2], step: usize) -> ([f64; 6], f64, f64) {
    let (sx, cx) = det_sincos(-TAU * k[0]);
    let (xc, yc) = (l.w as f64 / 2.0, l.h as f64 / 2.0);
    let row_sums = |y: usize| -> [f64; 6] {
        let ph = -TAU * (k[0] * 0.5 + k[1] * (y as f64 + 0.5));
        let (s0, c0) = det_sincos(ph);
        let (mut er, mut ei) = (c0, s0);
        let row = &l.px[y * l.stride..y * l.stride + l.w];
        let dy = y as f64 + 0.5 - yc;
        let (mut r, mut i, mut rx, mut ix) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (x, &v) in row.iter().enumerate() {
            let v: f64 = v.into();
            let dx = x as f64 + 0.5 - xc;
            r += v * er;
            i += v * ei;
            rx += v * dx * er;
            ix += v * dx * ei;
            let nr = er * cx - ei * sx;
            ei = er * sx + ei * cx;
            er = nr;
        }
        [r, i, rx, ix, r * dy, i * dy]
    };
    let ys: Vec<usize> = (0..l.h).step_by(step).collect();
    #[cfg(feature = "parallel")]
    let rows: Vec<[f64; 6]> = {
        use rayon::prelude::*;
        ys.par_iter().map(|&y| row_sums(y)).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<[f64; 6]> = ys.iter().map(|&y| row_sums(y)).collect();
    let mut s = [0.0f64; 6];
    for r in &rows {
        for j in 0..6 {
            s[j] += r[j];
        }
    }
    let var = |n: usize| (n * n) as f64 / 12.0;
    (s, var(l.w), var(l.h))
}

/// Refine a line frequency by the phase slope of the whole image (first-order:
/// for a line at k + D, Sx / S = 2 pi i D_x var_x, Sy / S likewise), then the
/// phase of the sum at the refined frequency, referred to X = 0.
fn refine_line<T: Px>(l: &Luma<T>, mut k: [f64; 2]) -> ([f64; 2], f64) {
    for it in 0..3 {
        let step = if it < 2 { 2 } else { 1 };
        let (s, vx, vy) = moment_sums(l, k, step);
        let n2 = s[0] * s[0] + s[1] * s[1];
        if n2 < 1e-12 {
            break;
        }
        // Im(Sx / S) = (S.re Sx.im - S.im Sx.re) / |S|^2.
        let qx = (s[0] * s[3] - s[1] * s[2]) / n2;
        let qy = (s[0] * s[5] - s[1] * s[4]) / n2;
        k = [k[0] + qx / (TAU * vx), k[1] + qy / (TAU * vy)];
    }
    let (s, _, _) = moment_sums(l, k, 1);
    (k, det_atan2(s[1], s[0]))
}

/// The encoder-to-received map of a candidate (refined, with translation).
pub fn checker_map<T: Px>(l: &Luma<T>, c: &Candidate) -> Option<GridMap> {
    let (k1, ph1) = refine_line(l, c.k1);
    let (k2, ph2) = refine_line(l, c.k2);
    // k1'.o = -arg S1 / 2 pi; the k2 line has a minus sign: k2'.o = -(arg S2 - pi) / 2 pi.
    let o1 = -ph1 / TAU;
    let o2 = -(ph2 - PI) / TAU;
    let det = k1[0] * k2[1] - k1[1] * k2[0];
    if det.abs() < 1e-12 {
        return None;
    }
    let ki = [[k2[1] / det, -k1[1] / det], [-k2[0] / det, k1[0] / det]];
    let o = [ki[0][0] * o1 + ki[0][1] * o2, ki[1][0] * o1 + ki[1][1] * o2];
    // Received frequencies K' = K A^-1 (rows), so A = K'^-1 K with
    // K = [(1, -1); (1, 1)] / 16.
    let kk = [[1.0 / 16.0, -1.0 / 16.0], [1.0 / 16.0, 1.0 / 16.0]];
    let a = [
        [ki[0][0] * kk[0][0] + ki[0][1] * kk[1][0], ki[0][0] * kk[0][1] + ki[0][1] * kk[1][1]],
        [ki[1][0] * kk[0][0] + ki[1][1] * kk[1][0], ki[1][0] * kk[0][1] + ki[1][1] * kk[1][1]],
    ];
    Some(GridMap { a, o, score: c.score })
}

/// Pair score at a known rotation (deg) and scale (measurement only).
#[doc(hidden)]
pub fn debug_checker_score<T: Px>(l: &Luma<T>, rot_deg: f64, scale: f64) -> f64 {
    match Spec::build(l) {
        Some(spec) => spec.pair(BASE / scale * spec.bins_per_cycle(), rot_deg.to_radians()),
        None => 0.0,
    }
}

impl GridMap {
    /// The same map rotated by `dtheta` (rad), scaled by `1 + ds` about the
    /// received point `c`, then shifted by `d` (received px).
    pub fn perturbed(&self, c: [f64; 2], dtheta: f64, ds: f64, d: [f64; 2]) -> GridMap {
        let (s, co) = det_sincos(dtheta);
        let f = 1.0 + ds;
        let r = [[f * co, -f * s], [f * s, f * co]];
        let mul = |m: [[f64; 2]; 2]| {
            [
                [r[0][0] * m[0][0] + r[0][1] * m[1][0], r[0][0] * m[0][1] + r[0][1] * m[1][1]],
                [r[1][0] * m[0][0] + r[1][1] * m[1][0], r[1][0] * m[0][1] + r[1][1] * m[1][1]],
            ]
        };
        let oc = [self.o[0] - c[0], self.o[1] - c[1]];
        let o = [
            c[0] + r[0][0] * oc[0] + r[0][1] * oc[1] + d[0],
            c[1] + r[1][0] * oc[0] + r[1][1] * oc[1] + d[1],
        ];
        GridMap { a: mul(self.a), o, score: self.score }
    }

    /// The encoder blocks (i0.., j0.., bw x bh) touching a w x h image.
    pub fn block_range(&self, w: usize, h: usize) -> (i64, i64, usize, usize) {
        let a = self.a;
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        let ai = [[a[1][1] / det, -a[0][1] / det], [-a[1][0] / det, a[0][0] / det]];
        let to_p = |x: f64, y: f64| {
            let (dx, dy) = (x - self.o[0], y - self.o[1]);
            [ai[0][0] * dx + ai[0][1] * dy, ai[1][0] * dx + ai[1][1] * dy]
        };
        let (w, h) = (w as f64, h as f64);
        let (mut pminx, mut pmaxx, mut pminy, mut pmaxy) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for c in [to_p(0.0, 0.0), to_p(w, 0.0), to_p(0.0, h), to_p(w, h)] {
            pminx = pminx.min(c[0]);
            pmaxx = pmaxx.max(c[0]);
            pminy = pminy.min(c[1]);
            pmaxy = pmaxy.max(c[1]);
        }
        let i0 = (pminx / 8.0).floor() as i64;
        let j0 = (pminy / 8.0).floor() as i64;
        let bw = ((pmaxx / 8.0).ceil() as i64 - i0).clamp(0, 8192) as usize;
        let bh = ((pmaxy / 8.0).ceil() as i64 - j0).clamp(0, 8192) as usize;
        (i0, j0, bw, bh)
    }
}

/// The 64 pixels of encoder block (i, j) (map-frame block coordinates),
/// bilinear-sampled at its pixel centres; `None` unless the block lies inside
/// the received image.
pub fn sample_block<T: Px>(l: &Luma<T>, m: &GridMap, i: i64, j: i64) -> Option<[f64; 64]> {
    let (w, h) = (l.w as f64, l.h as f64);
    let inside = |q: [f64; 2]| q[0] >= -0.5 && q[1] >= -0.5 && q[0] <= w + 0.5 && q[1] <= h + 0.5;
    let (pi, pj) = (i as f64 * 8.0, j as f64 * 8.0);
    if !(inside(m.apply([pi, pj]))
        && inside(m.apply([pi + 8.0, pj]))
        && inside(m.apply([pi, pj + 8.0]))
        && inside(m.apply([pi + 8.0, pj + 8.0])))
    {
        return None;
    }
    let mut px = [0.0f64; 64];
    for b in 0..8 {
        for a in 0..8 {
            let q = m.apply([pi + a as f64 + 0.5, pj + b as f64 + 0.5]);
            px[b * 8 + a] = l.bilinear(q[0] - 0.5, q[1] - 0.5).clamp(0.0, 255.0);
        }
    }
    Some(px)
}

/// Block averages (pixel mean - 128, the Fortress DC domain) and AC energies
/// of the encoder blocks touching the received image, sampled through `m`.
/// Field block (0, 0) is map-frame block (i0, j0); blocks not fully inside
/// the image are erased.
pub struct SampledBlocks {
    pub bw: usize,
    pub bh: usize,
    pub i0: i64,
    pub j0: i64,
    pub avg: Vec<f64>,
    pub energy: Vec<f64>,
    pub erased: Vec<bool>,
}

/// `energy_qt`: also measure the AC energies, quantized with this table
/// (otherwise `energy` is all zero; the pilot search needs only averages).
pub fn sample_blocks<T: Px>(l: &Luma<T>, m: &GridMap, energy_qt: Option<&[u16; 64]>) -> SampledBlocks {
    let (i0, j0, bw, bh) = m.block_range(l.w, l.h);
    let mut avg = vec![0.0f64; bw * bh];
    let mut energy = vec![0.0f64; bw * bh];
    let mut erased = vec![true; bw * bh];
    let row = |j: usize, avg: &mut [f64], energy: &mut [f64], erased: &mut [bool]| {
        for i in 0..bw {
            let Some(px) = sample_block(l, m, i0 + i as i64, j0 + j as i64) else { continue };
            avg[i] = px.iter().sum::<f64>() / 64.0 - 128.0;
            if let Some(qt) = energy_qt {
                let c = crate::codec::jpeg::pixels::dct_block_f64(&px, qt);
                let mut e = 0.0;
                for &v in &c[1..] {
                    if v.abs() >= 2 {
                        e += v as f64 * v as f64;
                    }
                }
                energy[i] = e;
            }
            erased[i] = false;
        }
    };
    let cw = bw.max(1);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        avg.par_chunks_mut(cw)
            .zip(energy.par_chunks_mut(cw))
            .zip(erased.par_chunks_mut(cw))
            .enumerate()
            .for_each(|(j, ((a, e), r))| row(j, a, e, r));
    }
    #[cfg(not(feature = "parallel"))]
    for (j, ((a, e), r)) in avg.chunks_mut(cw).zip(energy.chunks_mut(cw)).zip(erased.chunks_mut(cw)).enumerate() {
        row(j, a, e, r);
    }
    SampledBlocks { bw, bh, i0, j0, avg, energy, erased }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_whitening_matches_the_planes() {
        // Odd sizes, one shorter than the variance window, a row stride.
        for &(w, h) in &[(97usize, 61usize), (40, 23), (130, 140)] {
            let stride = w + 3;
            let px: Vec<f64> = (0..stride * h).map(|i| ((i * 7919) % 251) as f64 * 0.9 + (i % stride) as f64 * 0.2).collect();
            let l = Luma { px: &px, stride, w, h };
            let mut want: Vec<f64> = (0..h).flat_map(|y| px[y * stride..y * stride + w].to_vec()).collect();
            whiten_local(&mut want, w, h, 1);
            let got = whitened_full(&l).unwrap();
            for (i, (&g, &e)) in got.iter().zip(&want).enumerate() {
                assert!((g as f64 - e).abs() <= 1e-5 * e.abs().max(1.0), "{w}x{h} at {i}: {g} vs {e}");
            }
        }
    }

    /// A smooth random-ish luma with the checkerboard added, rotated and scaled.
    fn synth(w: usize, h: usize, rot_deg: f64, scale: f64, off: [f64; 2], amp: f64) -> Vec<f64> {
        let (s, c) = det_sincos(rot_deg.to_radians());
        let mut out = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                // Received pixel centre -> encoder frame p = A^-1 (X - o).
                let (dx, dy) = (x as f64 + 0.5 - off[0], y as f64 + 0.5 - off[1]);
                let px = (c * dx + s * dy) / scale;
                let py = (-s * dx + c * dy) / scale;
                let (bx, by) = ((px / 8.0).floor() as i64, (py / 8.0).floor() as i64);
                let chk = if (bx + by).rem_euclid(2) == 0 { amp } else { -amp };
                // Smooth shading plus white noise (an integer hash of the pixel).
                let hsh = ((x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
                    .wrapping_mul(0x1656_67B1_9E37_79F9);
                let noise = ((hsh >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 16.0;
                let content = 128.0 + 40.0 * det_sincos(px * 0.013).0 * det_sincos(py * 0.021).1 + noise;
                out[y * w + x] = content + chk;
            }
        }
        out
    }

    #[test]
    fn finds_rotation_scale_and_offset() {
        for &(rot, scale, off) in &[(0.0, 1.0, [3.0, 5.0]), (2.0, 1.0, [-11.0, 7.0]), (-4.0, 0.8, [5.5, 2.25])] {
            let (w, h) = (640, 480);
            let px = synth(w, h, rot, scale, off, 2.0);
            let l = Luma { px: &px, stride: w, w, h };
            let cands = checker_candidates(&l);
            assert!(!cands.is_empty(), "no candidate for rot {rot} scale {scale}");
            let m = checker_map(&l, &cands[0]).unwrap();
            let (r, s) = m.rotation_scale();
            assert!((r - rot).abs() < 0.05, "rot {r} vs {rot}");
            assert!((s - scale).abs() < 0.002, "scale {s} vs {scale}");
            // o must be a block corner of the true grid: A^-1 (o - off) on the 8-px lattice.
            let (sn, cs) = det_sincos(rot.to_radians());
            let (dx, dy) = (m.o[0] - off[0], m.o[1] - off[1]);
            let (px_, py_) = ((cs * dx + sn * dy) / scale, (-sn * dx + cs * dy) / scale);
            let frac = |v: f64| (v / 8.0 - (v / 8.0).round()).abs() * 8.0;
            assert!(frac(px_) < 0.3 && frac(py_) < 0.3, "corner off by ({px_}, {py_})");
        }
    }

    #[test]
    fn plain_image_has_no_candidate() {
        let (w, h) = (640, 480);
        let px = synth(w, h, 0.0, 1.0, [0.0, 0.0], 0.0);
        let l = Luma { px: &px, stride: w, w, h };
        assert!(checker_candidates(&l).is_empty());
    }
}
