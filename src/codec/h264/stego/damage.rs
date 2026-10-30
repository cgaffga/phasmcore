// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 stego damage estimator `D = E · T · X` (video stego v2).
//!
//! Spec: `docs/design/video/stego-cover-order-and-damage-costs.md` § 4–§ 6;
//! diagnosis: `docs/design/video/h264/intra-drift-diagnosis.md`.
//!
//! The encoder keeps predicting from its natural reconstruction while every
//! decoder sees the flipped one, so a coefficient flip's error is carried by
//! later prediction until the next key frame. The part that travels is the
//! **edge-constant** part of the flip's pixel delta: every intra predictor
//! and the inter interpolation filters have weights summing to one, so a
//! constant offset on the column / row a neighbour predicts from passes
//! through unchanged. This module prices that, from syntax alone:
//!
//! - **E** — energy of the flip's own pixel delta: `Δlevel = 1` through the
//!   flat-matrix dequant (`normAdjust4x4(QP % 6, i, j) << QP / 6`) and the
//!   H.264 inverse core transform, scaled by `|Δlevel|` (`2|c|` for a sign
//!   flip, `1` for a suffix-LSB flip). Exact rational arithmetic, kept as
//!   integers over a power-of-two denominator.
//! - **X** — spatial fan-out, `1 + κ_x · flood / E`, where `flood` is the
//!   edge-constant energy times the area that predicts from it. Only a
//!   coefficient with vertical frequency 0 leaves a constant right column,
//!   only one with horizontal frequency 0 a constant bottom row; every other
//!   AC basis has zero-mean edges. The area is the intra-reachable one: the
//!   8-connected intra MBs touching the block's MB or its downstream
//!   neighbours (right, below-left, below, below-right), capped by the
//!   right-and-below quadrant. In a key frame that is the quadrant itself;
//!   in a P frame inter MBs stop the spatial flood. Along the top MB row and
//!   the left MB column the offset is not diluted by a second neighbour, so
//!   that area counts [`DamageParams::edge_gain`] times.
//! - **T** — temporal fan-out `1 + κ_t · r`, `r` = share of the key-frame
//!   segment still to come after this frame.
//!
//! Everything is integer, from decoder-visible syntax only (position key,
//! clamped `|c|`, per-MB intra flag and QP, key-frame positions), so a shadow
//! decoder recomputes it bit for bit (§ 6). The primary uses the same value
//! as a cost multiplier (`content_costs` × `D / D_ref`).
//!
//! **Invariance for shadows** (§ 6): the inputs are unchanged by every
//! layer's flips. Sign flips keep `|c|`; a CoeffSuffixLsb flip changes `|c|`
//! by ±1 only at `|c| ≥ 16` and never below 16, so `|c|` is clamped at
//! [`SHADOW_MAG_CAP`] = 15 for the shadow features; MB types and QPs are not
//! touched by any stego layer (wire-only overrides change bypass bins only).

use super::hook::{BinKind, PositionKey, SyntaxPath};
use crate::codec::h264::tables::ZIGZAG_4X4;

/// Per-MB syntax the damage model reads, recorded by the CABAC walker.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MbSyntax {
    /// Walker frame index (VCL-NAL order, as in [`PositionKey::frame_idx`]).
    pub frame_idx: u32,
    pub mb_addr: u32,
    /// Luma QP of the MB (clamped to 0..=51).
    pub qp_y: u8,
    /// Chroma (Cb) QP of the MB, from `qp_y` and the PPS offset.
    pub qp_c: u8,
    /// [`MB_INTRA`] | [`MB_KEY`].
    pub flags: u8,
}

/// The walked stream's per-MB syntax, carried by `DomainCover::mb_syntax`.
#[derive(Clone, Debug, Default)]
pub struct MbSyntaxMap {
    /// Coded MB grid.
    pub mb_w: u32,
    pub mb_h: u32,
    /// One record per walked MB, in walk order.
    pub mbs: Vec<MbSyntax>,
}

/// The MB is intra-coded (I_NxN, I_16x16, I_PCM, SI).
pub const MB_INTRA: u8 = 1;
/// The MB belongs to an I / SI slice — a key frame (the GOP's IDR).
pub const MB_KEY: u8 = 2;

/// The calibrated κ's of `D = E · T · X` (spec § 4 / § 13). One set, used
/// by the primary costs and — fixed, since a shadow decoder recomputes it —
/// by the shadow buckets.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DamageParams {
    /// κ_x = `kx_num / kx_den`: weight of the flood energy against the
    /// flip's own energy.
    pub kx_num: u128,
    pub kx_den: u128,
    /// Gain on the reachable area along the top MB row / left MB column,
    /// where the propagated offset is not averaged with an undamaged second
    /// neighbour.
    pub edge_gain: u64,
    /// κ_t in sixteenths: `T = (16 + kt16 · remaining / segment) / 16`.
    pub kt16: u64,
}

/// The shipped κ's (calibration: `docs/design/video/h264/intra-drift-diagnosis.md`,
/// "v2 calibration" + the 2026-10-25 follow-up). Changing them changes which
/// positions the shadow layers use, i.e. the shadow wire format.
///
/// `kt16 = 16` is κ_t = 1 (maintainer decision, 2026-10-25): at 1080p it has
/// the best worst-MB minima of {0.5, 1, 2}, 0 flooded frames, and 1.5–2× less
/// temporal concentration than κ_t = 2. See the follow-up's κ_t table.
pub const DAMAGE_PARAMS: DamageParams = DamageParams { kx_num: 1, kx_den: 4, edge_gain: 4, kt16: 16 };

impl DamageParams {
    /// The primary's κ's: [`DAMAGE_PARAMS`], or for calibration sweeps
    /// `PHASM_DIAG_DAMAGE=kx_num/kx_den,edge_gain,kt16`. The primary decoder
    /// never reads costs, so this cannot break a decode; shadows always use
    /// [`DAMAGE_PARAMS`].
    pub fn primary() -> Self {
        let Ok(s) = std::env::var("PHASM_DIAG_DAMAGE") else { return DAMAGE_PARAMS };
        let p: Vec<&str> = s.split(',').collect();
        let kx: Vec<&str> = p.first().map(|t| t.split('/').collect()).unwrap_or_default();
        let num = |t: Option<&&str>, d: u128| t.and_then(|v| v.trim().parse().ok()).unwrap_or(d);
        DamageParams {
            kx_num: num(kx.first(), DAMAGE_PARAMS.kx_num),
            kx_den: num(kx.get(1), DAMAGE_PARAMS.kx_den).max(1),
            edge_gain: num(p.get(1), DAMAGE_PARAMS.edge_gain as u128) as u64,
            kt16: num(p.get(2), DAMAGE_PARAMS.kt16 as u128) as u64,
        }
    }
}

/// `|c|` cap for the shadow damage features (§ 6 invariance: CSL flips move
/// `|c|` by ±1 only at `|c| ≥ 16`, so every `|c| ≥ 15` must look alike).
pub const SHADOW_MAG_CAP: u16 = 15;
/// `|c|` cap for the primary (the J-UNIWARD term clamps at 32 too).
pub const PRIMARY_MAG_CAP: u16 = 64;

/// `normAdjust4x4(m, i, j)` (H.264 Table 8-15 / eq. 8-315): `v[m][0]` when
/// both indices are even, `v[m][1]` when both are odd, `v[m][2]` otherwise.
const NORM_ADJUST: [[u32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// `2 · Ci[n][k]` of the inverse core transform (spec 8.5.12.2: basis `k`
/// evaluated at sample `n`), so every entry is an integer.
const BASIS2: [[i32; 4]; 4] = [
    // n = 0     1     2     3        (rows: k)
    [2, 2, 2, 2],
    [2, 1, -1, -2],
    [2, -2, -2, 2],
    [1, -2, 2, -1],
];
/// `Σ_n BASIS2[k][n]²`.
const BASIS2_SQ: [u64; 4] = [16, 10, 16, 10];

fn norm_adjust(qp_rem: u32, i: usize, j: usize) -> u64 {
    let col = if i % 2 == 0 && j % 2 == 0 {
        0
    } else if i % 2 == 1 && j % 2 == 1 {
        1
    } else {
        2
    };
    NORM_ADJUST[qp_rem as usize][col] as u64
}

/// The exact unit table of § 5: `(scan position, QP % 6)` → pixel delta of
/// `Δlevel = 1` at `QP / 6 = 0`, in 1/256 px (`d · 2Ci[y][u] · 2Ci[x][v]`
/// with `d = normAdjust`). Shift by `QP / 6`, scale by `|Δlevel|`.
pub fn unit_delta_table() -> [[[[i32; 4]; 4]; 6]; 16] {
    let mut t = [[[[0i32; 4]; 4]; 6]; 16];
    for scan in 0..16 {
        let r = ZIGZAG_4X4[scan] as usize;
        let (u, v) = (r / 4, r % 4);
        for m in 0..6u32 {
            let na = norm_adjust(m, u, v) as i32;
            for y in 0..4 {
                for x in 0..4 {
                    t[scan][m as usize][y][x] = na * BASIS2[u][y] * BASIS2[v][x];
                }
            }
        }
    }
    t
}

/// Damage of a coefficient flip in units of `px² / 2^24` (energy × `T16` ×
/// `KX_DEN` over `65536`), before the per-frame terms. `None` for positions
/// the model does not price (DC, chroma DC, Intra16x16 luma DC, 8×8, MVD).
#[derive(Copy, Clone, Debug)]
struct CoeffTerms {
    /// `Δ² · nA² · B[u] · B[v]` — own energy · 65536, at QP / 6 = 0.
    e: u128,
    /// `Δ² · nA² · 4 · (b3[v]²·[u = 0] + b3[u]²·[v = 0])` — edge-constant
    /// energy per pixel · 65536.
    edge: u128,
    /// `QP / 6` of the plane.
    qp_div: u32,
}

fn coeff_terms(key: PositionKey, magnitude: u16, mag_cap: u16, qp_y: u8, qp_c: u8) -> Option<CoeffTerms> {
    let (scan, kind, qp) = match key.syntax_path() {
        SyntaxPath::Luma4x4 { coeff_idx, kind, .. } if coeff_idx != 0 => (coeff_idx, kind, qp_y),
        SyntaxPath::ChromaAc { coeff_idx, kind, .. } if coeff_idx != 0 => (coeff_idx, kind, qp_c),
        _ => return None,
    };
    let delta: u128 = match kind {
        BinKind::Sign => 2 * magnitude.clamp(1, mag_cap) as u128,
        BinKind::SuffixLsb => 1,
    };
    let r = ZIGZAG_4X4[scan as usize & 15] as usize;
    let (u, v) = (r / 4, r % 4);
    let qp = qp.min(51) as u32;
    let na = norm_adjust(qp % 6, u, v) as u128;
    let base = delta * delta * na * na;
    let e = base * (BASIS2_SQ[u] * BASIS2_SQ[v]) as u128;
    let b3 = |k: usize| (BASIS2[k][3] * BASIS2[k][3]) as u128;
    let mut edge_sq = 0u128;
    if u == 0 {
        edge_sq += b3(v); // constant right column
    }
    if v == 0 {
        edge_sq += b3(u); // constant bottom row
    }
    Some(CoeffTerms { e, edge: base * 4 * edge_sq, qp_div: qp / 6 })
}

/// Per-frame damage context.
#[derive(Clone, Debug, Default)]
struct FrameCtx {
    present: bool,
    /// Frames since the last key frame (0 = the key frame).
    key_pos: u32,
    /// Length of this key-frame segment (key frame + following non-key).
    seg_len: u32,
    qp_y: Vec<u8>,
    qp_c: Vec<u8>,
    /// Intra-reachable MB area per MB (before the edge gain).
    reach: Vec<u32>,
    /// Per-MB intra flag.
    intra: Vec<bool>,
}

/// Damage model for one walked stream (a GOP slab or a whole clip), built
/// from the walker's [`MbSyntax`] records (`DomainCover::mb_syntax`). Keys
/// are looked up by their own frame index, which the records share (a
/// remapped cover remaps both).
#[derive(Clone, Debug, Default)]
pub struct DamageModel {
    mb_w: u32,
    mb_h: u32,
    frames: Vec<FrameCtx>,
}

impl DamageModel {
    /// Build from walker records. `mb_w × mb_h` is the coded MB grid.
    /// The model of a walked cover.
    pub fn from_cover(cover: &super::inject::DomainCover) -> Self {
        let m = &cover.mb_syntax;
        Self::new(&m.mbs, m.mb_w, m.mb_h)
    }

    pub fn new(mbs: &[MbSyntax], mb_w: u32, mb_h: u32) -> Self {
        let _prof = crate::codec::h264::profile::scope("v2.damage_model");
        let n_mb = (mb_w as usize) * (mb_h as usize);
        let n_frames = mbs.iter().map(|m| m.frame_idx as usize + 1).max().unwrap_or(0);
        let mut frames: Vec<FrameCtx> = vec![FrameCtx::default(); n_frames];
        let mut intra: Vec<Vec<bool>> = vec![Vec::new(); n_frames];
        let mut is_key = vec![false; n_frames];
        for m in mbs {
            let f = m.frame_idx as usize;
            let fc = &mut frames[f];
            if !fc.present {
                fc.present = true;
                fc.qp_y = vec![0; n_mb];
                fc.qp_c = vec![0; n_mb];
                intra[f] = vec![false; n_mb];
            }
            let a = m.mb_addr as usize;
            if a < n_mb {
                fc.qp_y[a] = m.qp_y;
                fc.qp_c[a] = m.qp_c;
                intra[f][a] = m.flags & MB_INTRA != 0;
            }
            if m.flags & MB_KEY != 0 {
                is_key[f] = true;
            }
        }
        // Key-frame segments: position and length.
        let mut seg_start = 0usize;
        for f in 0..=n_frames {
            if f == n_frames || (f > seg_start && is_key[f]) {
                let len = (f - seg_start) as u32;
                for (k, fc) in frames[seg_start..f].iter_mut().enumerate() {
                    fc.key_pos = k as u32;
                    fc.seg_len = len.max(1);
                }
                seg_start = f;
            }
        }
        for f in 0..n_frames {
            if frames[f].present {
                frames[f].reach = reach_map(&intra[f], mb_w as usize, mb_h as usize);
                frames[f].intra = std::mem::take(&mut intra[f]);
            }
        }
        Self { mb_w, mb_h, frames }
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    fn frame(&self, key: PositionKey) -> Option<&FrameCtx> {
        self.frames.get(key.frame_idx() as usize).filter(|fc| fc.present)
    }

    /// The position's MB is intra-coded in a NON-key frame (an intra MB of
    /// a P frame). The damage-aware planner reads the share of a GOP's
    /// predicted flips that land there (`quality_predict`). `false` for a
    /// key outside the walk.
    pub(crate) fn intra_non_key(&self, key: PositionKey) -> bool {
        self.frame(key).is_some_and(|fc| {
            fc.key_pos > 0 && fc.intra.get(key.mb_addr() as usize).copied().unwrap_or(false)
        })
    }

    /// Damage `D` of flipping the bin at `key` (coefficient magnitude
    /// `magnitude`, clamped at `mag_cap`) under the shipped κ's, in
    /// `px² · 2^16 · 16 · kx_den` units. `None` for positions the model does
    /// not price — DC, chroma DC, Intra16x16 luma DC, 8×8, MVD — or a key
    /// outside the walk.
    pub fn damage(&self, key: PositionKey, magnitude: u16, mag_cap: u16) -> Option<u128> {
        self.damage_with(key, magnitude, mag_cap, &DAMAGE_PARAMS)
    }

    /// [`Self::damage`] under explicit κ's.
    pub fn damage_with(
        &self,
        key: PositionKey,
        magnitude: u16,
        mag_cap: u16,
        p: &DamageParams,
    ) -> Option<u128> {
        let fc = self.frame(key)?;
        let a = key.mb_addr() as usize;
        let (qp_y, qp_c) = (*fc.qp_y.get(a)?, *fc.qp_c.get(a)?);
        let t = coeff_terms(key, magnitude, mag_cap, qp_y, qp_c)?;
        let (mx, my) = (key.mb_addr() % self.mb_w.max(1), key.mb_addr() / self.mb_w.max(1));
        let gain = if mx == 0 || my == 0 { p.edge_gain } else { 1 };
        let area = fc.reach[a] as u128 * gain as u128;
        let x = t.e * p.kx_den + p.kx_num * t.edge * 256 * area;
        let remaining = (fc.seg_len - 1 - fc.key_pos.min(fc.seg_len - 1)) as u64;
        let t16 = 16 + (p.kt16 * remaining) / fc.seg_len as u64;
        Some((t16 as u128 * x) << (2 * t.qp_div))
    }

    /// The damage of a reference flip — `|c| = 1` sign flip, scan position 4
    /// (the first both-odd AC basis), no fan-out, `T = 1`, at luma QP `qp` —
    /// in the same units as [`Self::damage_with`] under `p`. The primary
    /// divides by it so `D` is ~1 for an ordinary late-GOP P-frame flip.
    pub fn reference(qp: i32, p: &DamageParams) -> u128 {
        let qp = qp.clamp(0, 51) as u32;
        let na = norm_adjust(qp % 6, 1, 1) as u128;
        let e = 4 * na * na * (BASIS2_SQ[1] * BASIS2_SQ[1]) as u128;
        (16 * e * p.kx_den) << (2 * (qp / 6))
    }

    pub fn mb_dims(&self) -> (u32, u32) {
        (self.mb_w, self.mb_h)
    }

    /// The capacity-policy predictor's view of one position
    /// (`stego::quality_predict`): `(D, persisted luma energy)`. `D` is
    /// [`Self::damage`] under the shipped κ's (the ranking the STC's costs
    /// follow, minus the content term); the energy is the flip's own luma
    /// pixel-delta energy times the frames left in its key-frame segment
    /// (the error persists until the next key frame), in `px² · 65536`, and
    /// 0 for a chroma position (the policy floors are luma PSNR). Integer,
    /// from syntax only. `None` where `damage` is `None`.
    pub(crate) fn predictor_terms(&self, key: PositionKey, magnitude: u16) -> Option<(u128, u128)> {
        let d = self.damage_with(key, magnitude, PRIMARY_MAG_CAP, &DAMAGE_PARAMS)?;
        if !matches!(key.syntax_path(), SyntaxPath::Luma4x4 { .. }) {
            return Some((d, 0));
        }
        let fc = self.frame(key)?;
        let a = key.mb_addr() as usize;
        let t = coeff_terms(key, magnitude, PRIMARY_MAG_CAP, *fc.qp_y.get(a)?, *fc.qp_c.get(a)?)?;
        let frames_left = (fc.seg_len - fc.key_pos.min(fc.seg_len - 1)) as u128;
        Some((d, (t.e << (2 * t.qp_div)) * frames_left))
    }

    /// The raw terms of [`Self::damage_with`] for one position, for
    /// calibration instruments. `None` exactly where `damage` is `None`.
    #[doc(hidden)]
    pub fn diag_terms(&self, key: PositionKey, magnitude: u16, mag_cap: u16) -> Option<DiagTerms> {
        let fc = self.frame(key)?;
        let a = key.mb_addr() as usize;
        let (qp_y, qp_c) = (*fc.qp_y.get(a)?, *fc.qp_c.get(a)?);
        let t = coeff_terms(key, magnitude, mag_cap, qp_y, qp_c)?;
        let (mx, my) = (key.mb_addr() % self.mb_w.max(1), key.mb_addr() / self.mb_w.max(1));
        let gain = if mx == 0 || my == 0 { DAMAGE_PARAMS.edge_gain } else { 1 };
        let scale = (1u128 << (2 * t.qp_div)) as f64 / 65536.0;
        Some(DiagTerms {
            e_px2: t.e as f64 * scale,
            edge_px2: t.edge as f64 * scale,
            area: (fc.reach[a] as u64 * gain) as u32,
            key_pos: fc.key_pos,
            seg_len: fc.seg_len,
            intra: fc.intra.get(a).copied().unwrap_or(false),
        })
    }
}

/// Calibration view of one position's damage terms ([`DamageModel::diag_terms`]).
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct DiagTerms {
    /// Own pixel-delta energy of the flip, px².
    pub e_px2: f64,
    /// Edge-constant energy per pixel, px².
    pub edge_px2: f64,
    /// Intra-reachable MB area incl. the top-row / left-column gain.
    pub area: u32,
    pub key_pos: u32,
    pub seg_len: u32,
    /// The position's MB is intra-coded.
    pub intra: bool,
}

/// Intra-reachable area per MB of one frame: the 8-connected intra
/// components touching the MB itself or its downstream neighbours (right,
/// below-left, below, below-right), summed over distinct components and
/// capped by the right-and-below quadrant. (The top-row / left-column gain
/// is applied at lookup, [`DamageParams::edge_gain`].)
fn reach_map(intra: &[bool], w: usize, h: usize) -> Vec<u32> {
    let n = w * h;
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let union = |p: &mut [u32], a: usize, b: usize| {
        let (ra, rb) = (find(p, a as u32), find(p, b as u32));
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            p[hi as usize] = lo;
        }
    };
    for y in 0..h {
        for x in 0..w {
            let m = y * w + x;
            if !intra[m] {
                continue;
            }
            // Forward 8-neighbours: right, below-left, below, below-right.
            if x + 1 < w && intra[m + 1] {
                union(&mut parent, m, m + 1);
            }
            if y + 1 < h {
                if x > 0 && intra[m + w - 1] {
                    union(&mut parent, m, m + w - 1);
                }
                if intra[m + w] {
                    union(&mut parent, m, m + w);
                }
                if x + 1 < w && intra[m + w + 1] {
                    union(&mut parent, m, m + w + 1);
                }
            }
        }
    }
    let mut size = vec![0u32; n];
    for m in 0..n {
        if intra[m] {
            let r = find(&mut parent, m as u32) as usize;
            size[r] += 1;
        }
    }
    let mut out = vec![0u32; n];
    for y in 0..h {
        for x in 0..w {
            let m = y * w + x;
            let mut roots = [u32::MAX; 5];
            let mut k = 0usize;
            let mut add = |p: &mut [u32], q: usize| {
                if intra[q] {
                    let r = find(p, q as u32);
                    if !roots[..k].contains(&r) {
                        roots[k] = r;
                        k += 1;
                    }
                }
            };
            add(&mut parent, m);
            if x + 1 < w {
                add(&mut parent, m + 1);
            }
            if y + 1 < h {
                if x > 0 {
                    add(&mut parent, m + w - 1);
                }
                add(&mut parent, m + w);
                if x + 1 < w {
                    add(&mut parent, m + w + 1);
                }
            }
            let total: u64 = roots[..k].iter().map(|&r| size[r as usize] as u64).sum();
            let quadrant = ((w - x) * (h - y)) as u64;
            out[m] = total.min(quadrant).min(u32::MAX as u64) as u32;
        }
    }
    out
}

/// Multiply the primary's coefficient costs by `D / D_ref` (spec § 4:
/// `cost = W · S · D`). Only finite costs change; wet positions stay wet, and
/// since `D ≥ 1` stays finite and positive, the ∞ pattern — all the capacity
/// probe reads — is unchanged. `qp` is the encode's QP (the reference
/// flip's plane QP). `PHASM_H264_DAMAGE_COSTS=0` skips it (A/B only).
pub fn apply_damage_costs(
    costs: &mut super::orchestrate::DomainCosts,
    cover: &super::inject::DomainCover,
    model: &DamageModel,
    qp: i32,
) {
    if std::env::var("PHASM_H264_DAMAGE_COSTS").as_deref() == Ok("0") {
        return;
    }
    let _prof = crate::codec::h264::profile::scope("v2.damage_costs");
    let p = DamageParams::primary();
    let d_ref = DamageModel::reference(qp, &p) as f64;
    for (bits, cv) in [
        (&cover.coeff_sign_bypass, &mut costs.coeff_sign_bypass),
        (&cover.coeff_suffix_lsb, &mut costs.coeff_suffix_lsb),
    ] {
        for (i, c) in cv.iter_mut().enumerate() {
            if !c.is_finite() {
                continue;
            }
            let (Some(&key), Some(&mag)) = (bits.positions.get(i), bits.magnitudes.get(i)) else {
                continue;
            };
            if let Some(d) = model.damage_with(key, mag, PRIMARY_MAG_CAP, &p) {
                let m = (d as f64 / d_ref).max(1e-6);
                *c = (*c as f64 * m) as f32;
            }
        }
    }
}

/// Shadow bucket of a damage value (spec § 6): `None` = excluded (more
/// than [`SHADOW_MAX_LOG2`] octaves over the reference flip), `Some(0)` =
/// the low-damage pool (at most [`SHADOW_LOW_LOG2`] octaves over it),
/// `Some(1)` = the rest. Integer only: `ilog2` of the damage relative to
/// [`DamageModel::reference`] at the fixed luma QP [`SHADOW_REF_QP`]
/// (absolute damage, so the same flip ranks the same in every clip).
///
/// **Two coarse buckets, on purpose.** Every shadow takes its positions in
/// `(bucket, keyed priority)` order, so with N ≥ 2 shadows all of them use
/// the SAME positions of every bucket they exhaust — full overlap, and the
/// earlier shadows lose half their bits there. Shadows only stay
/// independent random subsets (the collision model of
/// `shadow_capacity`) inside one bucket. So bucket 0 must hold the bulk of
/// the pool: on the corpus (720p, 30 frames, QP 26) `l ≤ 2` is 59–81 % of
/// the priced positions, the ordinary P-frame flips; the key-frame
/// edge-constant floods sit at `l` ≈ 7–23 and are excluded. The capacity
/// probe counts bucket 0 only (`GopCapacity::injectable_cover_bits`), so the
/// shown per-shadow figure keeps every shadow inside it.
pub fn shadow_bucket(d: u128) -> Option<u8> {
    let r = DamageModel::reference(SHADOW_REF_QP, &DAMAGE_PARAMS);
    let l = d.max(1).ilog2() as i64 - r.ilog2() as i64;
    if l > SHADOW_MAX_LOG2 {
        None
    } else if l <= SHADOW_LOW_LOG2 {
        Some(0)
    } else {
        Some(1)
    }
}

/// Fixed reference QP of the shadow damage scale (absolute damage).
pub const SHADOW_REF_QP: i32 = 26;
/// Shadow positions with `log2(D / D_ref(26)) >` this are excluded.
pub const SHADOW_MAX_LOG2: i64 = 6;
/// `log2(D / D_ref(26)) ≤` this is the low-damage bucket 0.
pub const SHADOW_LOW_LOG2: i64 = 2;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::stego::hook::EmbedDomain;

    fn key(frame: u32, mb: u32, coeff_idx: u8, kind: BinKind) -> PositionKey {
        PositionKey::new(
            frame,
            mb,
            if kind == BinKind::Sign { EmbedDomain::CoeffSignBypass } else { EmbedDomain::CoeffSuffixLsb },
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx, kind },
        )
    }

    /// The table reproduces the diagnosis example: QP 26, scan position 1
    /// (u = 0, v = 1), level 1 → sign flip Δ = 2 changes each row by
    /// +8 / +4 / −4 / −8, so the right column is a constant −8.
    #[test]
    fn unit_table_matches_the_measured_flood_flip() {
        let t = unit_delta_table();
        let qp = 26u32;
        let scale = 2 * (1 << (qp / 6)); // Δ = 2, << QP/6
        for y in 0..4 {
            let row: Vec<i32> = (0..4).map(|x| t[1][(qp % 6) as usize][y][x] * scale / 256).collect();
            assert_eq!(row, vec![8, 4, -4, -8], "row {y}");
        }
    }

    /// Exactly the vertical-frequency-0 bases have a constant right column,
    /// the horizontal-frequency-0 ones a constant bottom row; every other AC
    /// basis has zero-mean edges.
    #[test]
    fn edge_constant_iff_frequency_zero() {
        let t = unit_delta_table();
        for scan in 1..16 {
            let r = ZIGZAG_4X4[scan] as usize;
            let (u, v) = (r / 4, r % 4);
            let col: Vec<i32> = (0..4).map(|y| t[scan][0][y][3]).collect();
            let row: Vec<i32> = (0..4).map(|x| t[scan][0][3][x]).collect();
            assert_eq!(col.iter().all(|&c| c == col[0]), u == 0, "scan {scan} col");
            assert_eq!(row.iter().all(|&c| c == row[0]), v == 0, "scan {scan} row");
            if u != 0 {
                assert_eq!(col.iter().sum::<i32>(), 0);
            }
            if v != 0 {
                assert_eq!(row.iter().sum::<i32>(), 0);
            }
        }
    }

    /// `coeff_terms` agrees with the table: E = Σ (table · Δ)², edge = mean².
    #[test]
    fn terms_match_the_table() {
        let t = unit_delta_table();
        for scan in 1..16u8 {
            for qp in [20u8, 26, 33] {
                let k = key(0, 0, scan, BinKind::Sign);
                let ct = coeff_terms(k, 3, 64, qp, qp).unwrap();
                let tab = &t[scan as usize][(qp % 6) as usize];
                let e: u128 = tab.iter().flatten().map(|&v| (6 * v as i128).pow(2) as u128).sum();
                assert_eq!(ct.e, e, "scan {scan} qp {qp}");
                let rc = tab[0][3]; // constant right column if u == 0
                let bc = tab[3][0];
                let r = ZIGZAG_4X4[scan as usize] as usize;
                let mut edge = 0u128;
                if r / 4 == 0 {
                    edge += (6 * rc as i128).pow(2) as u128;
                }
                if r % 4 == 0 {
                    edge += (6 * bc as i128).pow(2) as u128;
                }
                // Both in px² · 65536: (Δ · v / 256)² · 65536 = (Δ · v)².
                assert_eq!(ct.edge, edge, "scan {scan} qp {qp} edge");
            }
        }
    }

    fn all_intra(w: u32, h: u32, frames: u32) -> Vec<MbSyntax> {
        let mut v = Vec::new();
        for f in 0..frames {
            for a in 0..w * h {
                v.push(MbSyntax {
                    frame_idx: f,
                    mb_addr: a,
                    qp_y: 26,
                    qp_c: 26,
                    flags: if f == 0 { MB_INTRA | MB_KEY } else { 0 },
                });
            }
        }
        v
    }

    /// In a key frame the edge-constant flip at the top-left MB carries the
    /// whole frame; the same flip deep in the frame, much less; a P-frame
    /// inter MB with no intra neighbour, none; zero-mean-edge bases, none.
    #[test]
    fn fan_out_ranks_the_flood_sources_first() {
        let (w, h) = (20u32, 12u32);
        let m = DamageModel::new(&all_intra(w, h, 10), w, h);
        let d = |f: u32, mb: u32, scan: u8| m.damage(key(f, mb, scan, BinKind::Sign), 1, 64).unwrap();
        let corner = d(0, 0, 1);
        let interior = d(0, 6 * w + 10, 1);
        let p_frame = d(5, 6 * w + 10, 1);
        let zero_mean = d(0, 0, 4); // scan 4 = raster 5: u = 1, v = 1
        assert!(corner > interior * 8, "corner {corner} interior {interior}");
        assert!(interior > p_frame * 8, "interior {interior} p {p_frame}");
        assert!(zero_mean * 100 < corner);
        // T: earlier frames of the segment weigh more.
        assert!(d(1, 6 * w + 10, 1) > d(9, 6 * w + 10, 1));
        // A sign flip of |c| = 2 has 4× the energy of |c| = 1.
        let k = key(5, 6 * w + 10, 4, BinKind::Sign);
        assert_eq!(m.damage(k, 2, 64).unwrap(), 4 * m.damage(k, 1, 64).unwrap());
        // DC / unpriced paths.
        assert!(m.damage(key(0, 0, 0, BinKind::Sign), 1, 64).is_none());
    }

    /// The shadow cap makes every |c| ≥ 15 alike, so a suffix-LSB flip of the
    /// same coefficient (|c| ± 1 at |c| ≥ 16) never moves a sign position's
    /// shadow features.
    #[test]
    fn shadow_features_ignore_csl_flips() {
        let (w, h) = (8u32, 8u32);
        let m = DamageModel::new(&all_intra(w, h, 4), w, h);
        let k = key(2, 9, 3, BinKind::Sign);
        for c in 16u16..200 {
            let a = m.damage(k, c, SHADOW_MAG_CAP);
            let b = m.damage(k, c + 1, SHADOW_MAG_CAP);
            assert_eq!(a, b);
            assert_eq!(a.map(shadow_bucket), m.damage(k, 15, SHADOW_MAG_CAP).map(shadow_bucket));
        }
    }

    /// P frame: inter MBs stop the spatial flood; an intra island floods only
    /// itself.
    #[test]
    fn p_frame_reach_is_the_connected_intra_area() {
        let (w, h) = (6usize, 4usize);
        let mut intra = vec![false; w * h];
        // A 2×2 intra island at (2,1)-(3,2).
        for (x, y) in [(2, 1), (3, 1), (2, 2), (3, 2)] {
            intra[y * w + x] = true;
        }
        let r = reach_map(&intra, w, h);
        assert_eq!(r[1 * w + 2], 4); // island MB: its component
        assert_eq!(r[0 * w + 2], 4); // top row, below = island (gain applied at lookup)
        assert_eq!(r[3 * w + 5], 0); // inter, nothing intra downstream
        assert_eq!(r[1 * w + 1], 4); // inter MB left of the island: its right neighbour
    }
}
