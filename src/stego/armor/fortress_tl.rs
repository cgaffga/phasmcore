// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Fortress TL: Fortress T with the payload on the five lowest AC
//! coefficients of every textured block, for messages longer than Fortress T
//! holds (about 550-1500 B at 1600 px).
//!
//! Same tile, pilots, header (format bit 1), checkerboard sync and decoder
//! geometry as [`fortress_tile`](super::fortress_tile); only the payload
//! differs. Design and measurements: `docs/design/image/armor-fortress-t.md`
//! (Fortress TL) and `eval/runs/2026-11-02-fortress-tl/RESULTS.md`.
//!
//! # Payload
//!
//! - Lanes AC01, AC10, AC11, AC02, AC20 of every block; no DC payload (the
//!   block mean carries only pilots, header and sync). Lane `l` of a block in
//!   tile cell of rank `k` is slot `l * T^2 + k` and carries coded bit
//!   `slot mod n` of the Fortress CC codeword.
//! - QIM step = [`AC_STEP`] x factor x strength:
//!   - the factor comes from the block-mean structure (resampling-stable, so
//!     the decoder rebuilds it after rotation or rescale), see [`activity`];
//!     blocks whose smallest directional activity is below [`SKIP`] (flat
//!     areas, and the smooth sides of straight edges) carry nothing;
//!   - the strength follows the message's copy class (a shorter message has
//!     more copies, each can be gentler); the class travels in the header's
//!     2 class bits.
//! - The encoder computes the factors from the cover alone (key-free), so the
//!   capacity is a property of the image and the HUD shows exactly what core
//!   writes ([`Plan`]).
//!
//! # Decoding
//!
//! The decoder rebuilds the factors from the received block means, leaving out
//! pilot and header blocks; blocks near the skip threshold count with a ramped
//! weight. Each lane's gain is estimated blindly (a resample blurs the AC
//! lanes, and a fixed-step lattice is gain-sensitive). On a transposed
//! orientation (quarter turn, transpose) AC01/AC10 and AC02/AC20 trade places;
//! signs do not matter (QIM's lattices are symmetric).

use crate::codec::jpeg::dct::DctGrid;
use crate::codec::jpeg::JpegImage;
use crate::stego::armor::fortress::{cc_tiers, dc_to_avg, qim_embed_avg, qim_extract_soft};

use super::fortress_tile::{coded_len, BlockField, TileKey, N_HDR, N_PILOT, T};

/// Natural-order indices of the AC lanes: (0,1), (1,0), (1,1), (0,2), (2,0).
pub const LANES: [usize; 5] = [1, 8, 9, 2, 16];
/// Lanes per block.
pub const N_LANES: usize = LANES.len();
/// Slots per tile (every cell, every lane).
pub const N_SLOTS: usize = N_LANES * T * T;
/// Base AC QIM step (orthonormal DCT units).
pub const AC_STEP: f64 = 42.0;
/// Step factor: linear from [`F_MIN`] at activity [`STEP_A0`] to [`F_MAX`] at
/// [`STEP_A1`] (gray levels). Busy blocks hide the larger steps; they also
/// carry the copies the skipped flat blocks cannot.
const STEP_A0: f64 = 2.0;
const STEP_A1: f64 = 10.0;
const F_MIN: f64 = 0.7;
const F_MAX: f64 = 2.0;
/// Smallest directional activity (gray levels) a block needs to carry lanes.
pub const SKIP: f64 = 1.5;
/// Decoder: half-width of the weight ramp around [`SKIP`].
const RAMP: f64 = 0.5;
/// Copy classes (r below 6, 6-11, 12-23, 24 and up) and their step strengths.
const CLASS_MIN_R: [usize; 4] = [0, 6, 12, 24];
const STRENGTH: [f64; 4] = [1.0, 0.95, 0.88, 0.8];
/// Minimum copies of the coded stream over the whole image.
pub const R_MIN: usize = 3;
/// Minimum copies' worth of usable blocks in every large crop window.
const R_CROP: f64 = 2.0;
/// The crop windows: this share (num / den) of each side, at the 4 corners
/// and the centre.
const CROP_NUM: usize = 7;
const CROP_DEN: usize = 10;
/// Blind gain search range (hundredths).
const GAIN_LO: u32 = 70;
const GAIN_HI: u32 = 105;

/// Per block: (activity for the step, activity for the skip decision).
///
/// The block means are smoothed by `[1 2 1]^2 / 16` (the same in all 8
/// orientations, and it cancels the checkerboard: 4 - 8 + 4 = 0). The step
/// activity is the mean `|S(n) - S(b)|` over the 8 neighbours (stable after a
/// resample); the skip activity is the smallest of the 4 directional means
/// (horizontal, vertical, 2 diagonals), so a block beside a straight edge
/// counts as flat: an edge does not mask noise in the smooth area next to
/// it, texture does. Excluded blocks (erased, or the decoder's pilot and
/// header blocks, whose QIM moves the mean) are left out; NaN where nothing
/// is known.
pub fn activity(avg: &[f64], excluded: &[bool], bw: usize, bh: usize) -> Vec<(f64, f64)> {
    let ok = |x: usize, y: usize| !excluded.get(y * bw + x).copied().unwrap_or(false);
    let mut sm = vec![f64::NAN; bw * bh];
    for y in 0..bh {
        for x in 0..bw {
            let (mut t, mut c) = (0.0, 0.0);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (u, v) = (x as i64 + dx, y as i64 + dy);
                    if u < 0 || v < 0 || u >= bw as i64 || v >= bh as i64 || !ok(u as usize, v as usize) {
                        continue;
                    }
                    let w = ((2 - dx.abs()) * (2 - dy.abs())) as f64;
                    t += w * avg[v as usize * bw + u as usize];
                    c += w;
                }
            }
            if c > 0.0 {
                sm[y * bw + x] = t / c;
            }
        }
    }
    let mut out = vec![(f64::NAN, f64::NAN); bw * bh];
    for y in 0..bh {
        for x in 0..bw {
            let s0 = sm[y * bw + x];
            if s0.is_nan() {
                continue;
            }
            let at = |dx: i64, dy: i64| -> Option<f64> {
                let (u, v) = (x as i64 + dx, y as i64 + dy);
                if u < 0 || v < 0 || u >= bw as i64 || v >= bh as i64 {
                    return None;
                }
                let sn = sm[v as usize * bw + u as usize];
                (!sn.is_nan()).then(|| (sn - s0).abs())
            };
            let (mut t, mut c) = (0.0, 0.0);
            let mut dir = f64::NAN;
            for (dx, dy) in [(1i64, 0i64), (0, 1), (1, 1), (1, -1)] {
                let (a, b) = (at(dx, dy), at(-dx, -dy));
                let v: Vec<f64> = [a, b].into_iter().flatten().collect();
                if v.is_empty() {
                    continue;
                }
                t += v.iter().sum::<f64>();
                c += v.len() as f64;
                let m = v.iter().sum::<f64>() / v.len() as f64;
                if dir.is_nan() || m < dir {
                    dir = m;
                }
            }
            if c > 0.0 {
                out[y * bw + x] = (t / c, dir);
            }
        }
    }
    out
}

/// Step factor of a block: 0 (no lanes) below [`SKIP`], else the mapping of
/// the step activity.
pub fn step_factor((a, s): (f64, f64)) -> f64 {
    if s.is_nan() || s < SKIP {
        return 0.0;
    }
    if a.is_nan() {
        return F_MIN;
    }
    F_MIN + (F_MAX - F_MIN) * ((a - STEP_A0) / (STEP_A1 - STEP_A0)).clamp(0.0, 1.0)
}

/// Decoder: (step factor, weight). Well below [`SKIP`] the block is not read;
/// within [`RAMP`] of it, read at the threshold's step with a weight ramping
/// 0 -> 1, so an encoder / decoder disagreement there is a weak vote.
fn decoder_factor((a, s): (f64, f64)) -> (f64, f64) {
    if s.is_nan() || s < SKIP - RAMP {
        return (0.0, 0.0);
    }
    (step_factor((a, s.max(SKIP))), ((s - (SKIP - RAMP)) / (2.0 * RAMP)).clamp(0.0, 1.0))
}

/// Copy class of `r` copies (the header's 2 class bits).
pub fn class(r: usize) -> usize {
    CLASS_MIN_R.iter().rposition(|&m| m <= r).unwrap_or(0)
}

fn strength(rc: usize) -> f64 {
    STRENGTH[rc.min(3)]
}

/// The key-free layout of a cover: per block the step factor (0 = no lanes),
/// computed from the cover's own block means. `armor_capacity_info` and the
/// encoder both use it, so the Fortress maximum the HUD shows is the one the
/// encoder applies.
pub struct Plan {
    pub bw: usize,
    pub bh: usize,
    pub factors: Vec<f64>,
    /// Blocks that carry lanes.
    pub usable: usize,
    /// The fewest usable blocks in any crop window.
    pub min_window: usize,
}

impl Plan {
    pub fn new(img: &JpegImage) -> Plan {
        let (bw, bh) = super::pipeline::real_luma_blocks(img);
        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let qt_dc = img.quant_table(qt_id).map_or(1, |q| q.values[0]);
        let g = img.dct_grid(0);
        let avg: Vec<f64> = (0..bw * bh).map(|i| dc_to_avg(g.get(i / bw, i % bw, 0, 0), qt_dc)).collect();
        let factors: Vec<f64> = activity(&avg, &[], bw, bh).into_iter().map(step_factor).collect();
        let usable = factors.iter().filter(|&&f| f > 0.0).count();
        let (cw, ch) = (bw * CROP_NUM / CROP_DEN, bh * CROP_NUM / CROP_DEN);
        let count = |x0: usize, y0: usize| -> usize {
            (y0..y0 + ch).map(|y| (x0..x0 + cw).filter(|&x| factors[y * bw + x] > 0.0).count()).sum()
        };
        let (mx, my) = (bw - cw, bh - ch);
        let min_window = [(0, 0), (mx, 0), (0, my), (mx, my), (mx / 2, my / 2)]
            .iter()
            .map(|&(x, y)| count(x, y))
            .min()
            .unwrap_or(0);
        Plan { bw, bh, factors, usable, min_window }
    }

    /// Whether a codeword of `n` bits fits: within one tile's slots, at least
    /// [`R_MIN`] copies over the image, and [`R_CROP`] copies' worth in every
    /// crop window (else a large crop could leave too few; the message then
    /// goes to Standard).
    pub fn fits(&self, n: usize) -> bool {
        super::fortress_tile::fortress_size_ok(self.bw, self.bh)
            && n <= N_SLOTS
            && N_LANES * self.usable / n >= R_MIN
            && (N_LANES * self.min_window) as f64 / n as f64 >= R_CROP
    }

    /// Largest Fortress TL frame (bytes) this cover carries, for any passphrase.
    pub fn max_frame_bytes(&self) -> usize {
        cc_tiers().into_iter().filter(|&t| self.fits(coded_len(t))).max().unwrap_or(0)
    }

    /// Copies of an `n`-bit codeword.
    pub fn copies(&self, n: usize) -> usize {
        N_LANES * self.usable / n
    }
}

/// Embed the coded stream on the AC lanes (after every block mean is final).
pub fn embed(grid: &mut DctGrid, qtv: &[u16; 64], key: &TileKey, plan: &Plan, coded: &[u8], n: usize, rc: usize) {
    let g = strength(rc);
    for by in 0..plan.bh {
        for bx in 0..plan.bw {
            let f = plan.factors[by * plan.bw + bx];
            if f == 0.0 {
                continue;
            }
            let k = key.rank_at(bx, by);
            for (l, &idx) in LANES.iter().enumerate() {
                let q = qtv[idx] as f64;
                let v = grid.get(by, bx, idx / 8, idx % 8) as f64 * q;
                let nv = qim_embed_avg(v, AC_STEP * f * g, coded[(l * T * T + k) % n]);
                grid.set(by, bx, idx / 8, idx % 8, (nv / q).round().clamp(-2047.0, 2047.0) as i16);
            }
        }
    }
}

/// Soft values of the `n` coded bits from a field at tile offset (`ox`, `oy`)
/// (`key` already oriented), header copy class `rc`; also the number of lane
/// values read.
pub fn soft(field: &BlockField, key: &TileKey, ox: usize, oy: usize, n: usize, rc: usize) -> (Vec<f64>, usize) {
    let mut soft = vec![0.0f64; n];
    if field.ac.is_empty() {
        return (soft, 0);
    }
    let rank_of = |i: usize| key.rank_at(i % field.bw + ox, i / field.bw + oy);
    let g = strength(rc);
    // Received lane per encoder lane.
    let lp: [usize; 5] = if key.transposed { [1, 0, 2, 4, 3] } else { [0, 1, 2, 3, 4] };
    let excluded: Vec<bool> =
        (0..field.bw * field.bh).map(|i| field.erased[i] || rank_of(i) < N_PILOT + N_HDR).collect();
    let fw: Vec<(f64, f64)> = activity(&field.avg, &excluded, field.bw, field.bh)
        .into_iter()
        .enumerate()
        .map(|(i, a)| if field.erased[i] { (0.0, 0.0) } else { decoder_factor(a) })
        .collect();
    // Blind per-lane gain: the gain whose rescaled values sit closest to the
    // lattice (mean |soft| largest).
    let mut gain = [1.0f64; N_LANES];
    for (l, gl) in gain.iter_mut().enumerate() {
        let mut best = (1.0f64, f64::MIN);
        for gi in GAIN_LO..=GAIN_HI {
            let gg = gi as f64 / 100.0;
            let mut sc = 0.0f64;
            for (i, &(f, _)) in fw.iter().enumerate() {
                if f == 0.0 {
                    continue;
                }
                let step = AC_STEP * f * g;
                sc += (qim_extract_soft(field.ac[i][lp[l]] / gg, step) / (step / 2.0)).abs();
            }
            if sc > best.1 {
                best = (gg, sc);
            }
        }
        *gl = best.0;
    }
    let mut used = 0usize;
    for (i, &(f, w)) in fw.iter().enumerate() {
        if f == 0.0 || w == 0.0 {
            continue;
        }
        let k = rank_of(i);
        let step = AC_STEP * f * g;
        for l in 0..N_LANES {
            let sv = qim_extract_soft(field.ac[i][lp[l]] / gain[l], step) / (step / 2.0);
            soft[(l * T * T + k) % n] += sv * w;
            used += 1;
        }
    }
    (soft, used)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The activity is the same in every orientation of the field (the
    /// decoder reads turned and mirrored images), up to the orientation.
    #[test]
    fn activity_is_orientation_invariant() {
        let (w, h) = (23usize, 17usize);
        let avg: Vec<f64> = (0..w * h).map(|i| ((i * 7919) % 97) as f64 * 0.3 + (i % w) as f64).collect();
        let a = activity(&avg, &[], w, h);
        // Mirror in x.
        let m: Vec<f64> = (0..w * h).map(|i| avg[(i / w) * w + (w - 1 - i % w)]).collect();
        let am = activity(&m, &[], w, h);
        // Transpose.
        let t: Vec<f64> = (0..w * h).map(|i| avg[(i % h) * w + i / h]).collect();
        let at = activity(&t, &[], h, w);
        for y in 0..h {
            for x in 0..w {
                let p = a[y * w + x];
                let q = am[y * w + (w - 1 - x)];
                let r = at[x * h + y];
                assert!((p.0 - q.0).abs() < 1e-9 && (p.1 - q.1).abs() < 1e-9, "mirror at {x},{y}");
                assert!((p.0 - r.0).abs() < 1e-9 && (p.1 - r.1).abs() < 1e-9, "transpose at {x},{y}");
            }
        }
    }

    /// A flat field carries nothing; a straight edge in a flat field carries
    /// nothing either (its smooth sides are not texture); texture does.
    #[test]
    fn edges_are_not_texture() {
        let (w, h) = (20usize, 20usize);
        let flat = vec![10.0f64; w * h];
        assert!(activity(&flat, &[], w, h).into_iter().all(|a| step_factor(a) == 0.0));
        let edge: Vec<f64> = (0..w * h).map(|i| if i / w < 10 { 0.0 } else { 80.0 }).collect();
        assert!(activity(&edge, &[], w, h).into_iter().all(|a| step_factor(a) == 0.0));
        let tex: Vec<f64> = (0..w * h).map(|i| ((i * 2654435761usize) % 61) as f64).collect();
        let f: Vec<f64> = activity(&tex, &[], w, h).into_iter().map(step_factor).collect();
        assert!(f.iter().filter(|&&v| v > 0.0).count() > w * h * 3 / 4);
    }
}
