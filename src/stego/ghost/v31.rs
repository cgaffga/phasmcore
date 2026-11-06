// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Ghost embedding v3.1: ternary STC over **all** Y AC coefficients, the way
//! reference J-UNIWARD embeds. Design and evidence:
//! `docs/design/image/ghost-embedding-v3.md` (v3.1 section). v3.1 replaced
//! v3 (core 0.5 / apps 1.9) for encoding; v3 images still decode
//! ([`super::v3_legacy`], decode-only).
//!
//! - **Cover** = every Y AC coefficient, zeros included, in a keyed
//!   permutation (the v3 order). The set never changes, so no direction is
//!   forbidden and the decoder needs no costs at all.
//! - **Symbols** = two bit planes of the coefficient value: `P1 = c mod 2`
//!   (parity) and `P2 = ⌊(c + s)/2⌋ mod 2` with a keyed bit `s` per
//!   coefficient ([`PairKey`]). Every ±1 change flips P1; exactly one of the
//!   two directions also flips P2 (`+1` when `c + s` is odd, `−1` when
//!   even). With `s` keyed, which pair `{2k, 2k+1}` or `{2k−1, 2k}` a value
//!   belongs to is secret, so an excess of either layer's changes has no
//!   directional signature (v3: `s = 0`, R3.16).
//! - **Ternary embedding** = the double-layered STC construction (Filler,
//!   Judas, Fridrich 2011, §VI): layer 2 embeds its bits into P2 with costs
//!   from the marginal P2-flip probabilities of the optimal ternary
//!   embedding; layer 1 embeds the rest into P1, where layer-2 flips are
//!   fixed and every other change takes the P2-preserving direction.
//! - **Codes** = mixed-width STCs ([`trellis::Code`]): 48 header blocks of a
//!   ladder width (the decoder finds the frame length there), then blocks of
//!   `w` and `w + 1` positions so every position of the lane is used (v3:
//!   one ladder width, 0–17 % of a lane idle, R3.18).
//! - **Costs** (encoder only) = J-UNIWARD on the unrounded decompressed
//!   image (symmetric ρ⁺ = ρ⁻) or, with side information, `ρ·(1 − 2|e|)`
//!   toward the precover and `ρ` away when it is exact (Deep Cover: core
//!   wrote the cover from the same pixels; ties keep `ρ`), or
//!   `ρ·max(1 − 2|e|, 0.3)` toward the precover when approximate.
//! - **Lanes**: covers of more than 2²⁰ AC coefficients are cut into
//!   independent STC pairs (lanes) that are planned in parallel.
//!
//! The H-hat and pair seeds are domain-separated from v3's and v1's
//! (SHA-256 of a label and the structural key), so a v3.1 image never
//! parses as v3 or v1 or vice versa (the AES tag and CRC are the
//! validators). The order is v3's on purpose: one shuffle per decode reads
//! both generations ([`decode_all`]).

use crate::codec::jpeg::JpegImage;
use crate::codec::jpeg::dct::DctGrid;
use crate::det_math::{det_exp, det_ln};
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload::PayloadData;
use crate::stego::progress;
use crate::stego::side_info::SideInfo;
use crate::stego::stc::trellis::Code;
use crate::stego::stc::{hhat, trellis};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};

/// STC constraint height of every v3.1 trellis (primary layers and
/// shadows). Part of the wire format; v3's height (frozen at 9 on
/// 2026-10-27 after the height sweep, `ghost-embedding-v3.md`, "STC
/// constraint height").
pub const V31_STC_H: usize = 9;

/// Header blocks of every layer code: the first `min(HEADER_BLOCKS, m)`
/// blocks have the ladder width ([`choose_w`]), so the decoder can read the
/// frame length (48 bits) before it knows the frame length. Part of the
/// wire format.
pub const HEADER_BLOCKS: usize = 48;

/// Largest |coefficient| a baseline 8-bit JPEG AC coefficient may take.
const COEFF_MAX: i16 = 1023;
/// SI cost floor (as `cost::uniward::MIN_SI_COST_F32`).
const MIN_SI_COST: f32 = 1e-6;
/// Positions sampled for the payload split (λ search).
const LAMBDA_SAMPLES: usize = 1 << 15;

/// Domain-separated seeds.
pub struct V31Seeds {
    /// The keyed order (v3's label: shared on purpose).
    pub perm: [u8; 32],
    pub hhat1: [u8; 32],
    pub hhat2: [u8; 32],
    /// The pair bits `s` ([`PairKey`]).
    pub pair: [u8; 32],
}

fn label_hash(label: &[u8], key: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(label);
    h.update(key);
    h.finalize().into()
}

pub fn v31_seeds(structural_key: &[u8; 64]) -> V31Seeds {
    V31Seeds {
        perm: label_hash(b"phasm-ghost-v3/perm\0", structural_key),
        hhat1: label_hash(b"phasm-ghost-v3.1/hhat1", structural_key),
        hhat2: label_hash(b"phasm-ghost-v3.1/hhat2", structural_key),
        pair: label_hash(b"phasm-ghost-v3.1/pair", structural_key),
    }
}

// ---------------------------------------------------------------------------
// Width ladder, H-hat, layer codes
// ---------------------------------------------------------------------------

/// The header widths of v3.1 and the shadow widths of v3 and v3.1 (wire,
/// unchanged from v3): 1..=16, then ×≈1.2 steps.
pub(crate) fn ladder() -> impl Iterator<Item = usize> {
    let mut w = 1usize;
    std::iter::from_fn(move || {
        let cur = w;
        w = if w < 16 { w + 1 } else { w + w / 5 };
        Some(cur)
    })
}

/// The widest ladder width with `m · w ≤ n`; `None` if `m == 0` or `m > n`.
pub fn choose_w(n: usize, m: usize) -> Option<usize> {
    if m == 0 || m > n {
        return None;
    }
    ladder().take_while(|&w| m.saturating_mul(w) <= n).last()
}

/// The STC submatrix of a v3.1 trellis: random columns with the first and
/// last bit set (the standard STC construction; plain random odd-weight
/// columns cost ~20 % more distortion — `stc_coding_loss`).
pub fn v31_hhat(w: usize, seed: &[u8; 32], h: usize) -> Vec<Vec<u32>> {
    hhat::generate_hhat_end_set(h, w, seed)
}

/// Packed H-hat columns (bit `r` = row `r`).
pub(crate) fn packed_cols(hh: &[Vec<u32>], w: usize) -> Vec<u32> {
    (0..w).map(|c| hhat::column_packed(hh, c)).collect()
}

/// Block structure of one layer of a lane: `m` bits over `n` positions,
/// the first `min(HEADER_BLOCKS, m)` blocks of the ladder width
/// `choose_w(n, m)`, the other blocks sharing the remaining positions evenly
/// (widths `short_w` and `short_w + 1`). Every position is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerShape {
    pub m: usize,
    pub head: usize,
    pub head_w: usize,
    pub rest_n: usize,
    pub short_w: usize,
}

impl LayerShape {
    /// `None` if `m == 0` or `m > n`.
    pub fn new(n: usize, m: usize) -> Option<Self> {
        let head_w = choose_w(n, m)?;
        let head = HEADER_BLOCKS.min(m);
        let rest_n = n - head * head_w;
        let short_w = if m > head { rest_n / (m - head) } else { 0 };
        Some(Self { m, head, head_w, rest_n, short_w })
    }

    /// Positions the layer's trellis walks (the whole lane when it carries
    /// more than the header).
    pub fn n(&self) -> usize {
        self.head * self.head_w + if self.m > self.head { self.rest_n } else { 0 }
    }

    /// Average block width (memory and progress estimates).
    pub fn avg_w(&self) -> usize {
        self.n().div_ceil(self.m.max(1)).max(1)
    }

    /// The layer's code with its H-hat columns (`seed`: the layer's H-hat
    /// seed; one H-hat per width).
    pub fn code(&self, seed: &[u8; 32], h: usize) -> Code {
        let cols = |w: usize| -> Vec<u32> { if w == 0 { Vec::new() } else { packed_cols(&v31_hhat(w, seed, h), w) } };
        let rm = self.m - self.head;
        let long = if rm > 0 && self.rest_n % rm != 0 { cols(self.short_w + 1) } else { Vec::new() };
        Code::new(self.m, self.head, cols(self.head_w), if rm > 0 { self.rest_n } else { 0 }, cols(self.short_w), long)
            .expect("a layer shape gives a consistent code")
    }
}

// ---------------------------------------------------------------------------
// Positions
// ---------------------------------------------------------------------------

/// Every Y AC flat index (`block · 64 + k`, k = 1..63), raster order.
pub fn ac_flat_indices(grid: &DctGrid) -> Vec<u32> {
    let nb = grid.blocks_wide() * grid.blocks_tall();
    let mut out = Vec::with_capacity(nb * 63);
    for b in 0..nb as u32 {
        for k in 1..64u32 {
            out.push(b * 64 + k);
        }
    }
    out
}

/// Number of Y AC coefficients.
pub fn ac_count(grid: &DctGrid) -> usize {
    grid.blocks_wide() * grid.blocks_tall() * 63
}

pub use crate::stego::permute::shuffle_u32;

/// The keyed embedding order: all AC flat indices, permuted.
pub fn order(grid: &DctGrid, seeds: &V31Seeds) -> Vec<u32> {
    let mut o = ac_flat_indices(grid);
    shuffle_u32(&mut o, &seeds.perm);
    o
}

#[inline]
fn coeff(grid: &DctGrid, fi: u32) -> i16 {
    grid.coeffs()[fi as usize]
}

/// The keyed pair bit `s` of every coefficient of the Y grid, by flat index:
/// bit `fi` of the ChaCha20 keystream of the pair seed (byte `fi / 8`, bit
/// `fi mod 8`). Part of the wire format.
pub struct PairKey {
    words: Vec<u64>,
}

impl PairKey {
    pub fn new(seed: &[u8; 32], total_coefficients: usize) -> Self {
        use crate::stego::rng_compat::WordRng;
        use rand::SeedableRng;
        // The raw keystream (no `rand` sampler involved; pinned by
        // `rng_golden_tests`, `ghost_v31_pairs`).
        let mut bytes = vec![0u8; total_coefficients.div_ceil(64) * 8];
        WordRng::fill_bytes(&mut rand_chacha::ChaCha20Rng::from_seed(*seed), &mut bytes);
        Self { words: bytes.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect() }
    }

    /// `s` of coefficient `fi` (0 or 1).
    #[inline]
    pub fn get(&self, fi: u32) -> i16 {
        ((self.words[fi as usize >> 6] >> (fi & 63)) & 1) as i16
    }
}

#[inline]
pub fn plane1(c: i16) -> u8 {
    (c & 1) as u8
}

/// Bit plane 2 of a coefficient value `c` with pair bit `s`: `⌊(c + s)/2⌋ mod 2`.
#[inline]
pub fn plane2(c: i16, s: i16) -> u8 {
    // Wrapping: a crafted JPEG can hold |c| = 32767; the wrap moves the sum by
    // 2^16, which leaves both bits read here unchanged.
    ((c.wrapping_add(s) >> 1) & 1) as u8
}

/// The direction (+1/−1) of the ±1 change that also flips P2.
#[inline]
fn dir2(c: i16, s: i16) -> i16 {
    if c.wrapping_add(s) & 1 != 0 { 1 } else { -1 }
}


/// A set of flat coefficient indices that must not change (positions owned
/// by shadow layers). One bit per coefficient.
pub struct WetMask {
    bits: Vec<u64>,
}

impl WetMask {
    pub fn new(total_coefficients: usize) -> Self {
        Self { bits: vec![0; total_coefficients.div_ceil(64)] }
    }

    #[inline]
    pub fn set(&mut self, fi: u32) {
        self.bits[fi as usize >> 6] |= 1 << (fi & 63);
    }

    #[inline]
    pub fn get(&self, fi: u32) -> bool {
        (self.bits[fi as usize >> 6] >> (fi & 63)) & 1 == 1
    }
}

/// Luma blocks of the image itself: `(ceil(Wy / 8), ceil(Hy / 8))` with the
/// luma plane's own size `ceil(W · h / hmax)` x `ceil(H · v / vmax)` (T.81 A.1.1).
/// The coefficient grid can be larger: it is padded to whole MCUs.
pub fn real_luma_blocks(img: &JpegImage) -> (usize, usize) {
    let fi = img.frame_info();
    let c = &fi.components[0];
    let comp = |size: u16, f: u8, max: u8| (size as usize * f as usize).div_ceil(max.max(1) as usize);
    (
        comp(fi.width, c.h_sampling, fi.max_h_sampling).div_ceil(8),
        comp(fi.height, c.v_sampling, fi.max_v_sampling).div_ceil(8),
    )
}

/// Every position of the luma blocks outside the image ("dummy" blocks: the
/// MCU padding of e.g. a 4:2:0 image with an odd block count), or `None` when
/// there are none. libjpeg writes them with all AC zero (DC copied from the
/// neighbour), so a non-zero AC there is a deterministic stego tell (Benes,
/// Harmann & Boehme, "Steganalysis Exploiting JPEG Block Padding", WIFS 2026).
/// Every encoder treats them as wet: never changed, still read by the
/// decoder (no wire change). Other encoders may put content there; it is
/// left untouched as well.
pub fn dummy_block_wet(img: &JpegImage) -> Option<WetMask> {
    let grid = img.dct_grid(0);
    let (bw, bt) = (grid.blocks_wide(), grid.blocks_tall());
    let (rw, rt) = real_luma_blocks(img);
    if rw >= bw && rt >= bt {
        return None;
    }
    let mut wet = WetMask::new(grid.coeffs().len());
    for br in 0..bt {
        for bc in 0..bw {
            if br >= rt || bc >= rw {
                let base = ((br * bw + bc) * 64) as u32;
                for k in 0..64 {
                    wet.set(base + k);
                }
            }
        }
    }
    Some(wet)
}

/// [`WetMask::new`] for `img`'s luma grid with the dummy blocks already wet.
pub fn wet_mask_for(img: &JpegImage) -> WetMask {
    dummy_block_wet(img).unwrap_or_else(|| WetMask::new(img.dct_grid(0).coeffs().len()))
}

// ---------------------------------------------------------------------------
// Costs and the payload split
// ---------------------------------------------------------------------------

/// What an encode did (quality scoring and measurement).
#[derive(Debug, Clone, Default)]
pub struct V31Stats {
    pub n: usize,
    pub m: usize,
    /// Lanes used.
    pub lanes: usize,
    /// Layer-2 bits of lane 0.
    pub m2: usize,
    pub w1: usize,
    pub w2: usize,
    pub changes: usize,
    /// Changes made by layer 2 (P2 flips); the rest are layer 1's.
    pub changes2: usize,
    /// Σ ρ of the applied changes (direction-specific cost).
    pub distortion: f64,
    pub median_cost: f32,
    /// Rate-distortion bound of the optimal ternary embedding of the payload,
    /// estimated on the λ subsample (measurement only, see
    /// [`EncodeOpts::bound`]).
    pub bound_distortion: f64,
}

/// Side information of one coefficient as the cost function sees it.
#[derive(Clone, Copy)]
pub struct SiTerm {
    /// Rounding error `e = precover − c` in `[−½, ½]`.
    pub e: f32,
    /// `e` is the cover encoder's own rounding (phasm wrote the cover).
    pub exact: bool,
    /// Exact `|e| = ½` tie.
    pub tie: bool,
}

#[inline]
pub fn si_term(si: Option<&SideInfo>, fi: usize) -> Option<SiTerm> {
    si.map(|s| SiTerm { e: s.error_at(fi), exact: s.is_exact(), tie: s.is_tie(fi) })
}

/// Floor of the SI factor `1 − 2|e|` for *approximate* side information.
const SI_FACTOR_FLOOR_APPROX: f32 = 0.3;

/// Per-position ±1 costs `(ρ⁺, ρ⁻)`.
///
/// *Approximate* side information (the float DCT of the source pixels while
/// the cover came from another encoder — platform / `image` crate, integer
/// DCTs): the SI factor `1 − 2|e|` is floored at 0.3, because `|e|` near ½ is
/// unreliable and an unfloored factor makes those coefficients nearly free —
/// measured: GFR→FLD P_E 0.32 → 0.45 and histogram P_E 0.27 → 0.44 with the
/// floor (`eval/runs/2026-10-26-ghost-structural/RESULTS.md` §7).
///
/// *Exact* side information (phasm wrote the cover,
/// `SideInfo::compute_exact`): SI-UNIWARD's `ρ(1 − 2|e|)` toward the
/// precover, no floor, `ρ` away; exact ties `|e| = ½` are unmodulated so the
/// cover's tie-break (half away from zero) is kept — freeing them was
/// measured worse (Δ P_E −0.040 at 0.20), as were an away cost `ρ(1 + 2|e|)`
/// and a 0.3 floor (`eval/runs/2026-10-27-deep-cover-exact-si/RESULTS.md` §3).
#[inline]
fn pm_costs(rho: f32, c: i16, si: Option<SiTerm>) -> (f32, f32) {
    let (mut rp, mut rm) = match si {
        None => (rho, rho),
        Some(t) if t.exact && t.tie => (rho, rho),
        Some(t) => {
            let factor = 1.0 - 2.0 * t.e.abs();
            let factor = if t.exact { factor } else { factor.max(SI_FACTOR_FLOOR_APPROX) };
            let toward = (rho * factor).max(MIN_SI_COST);
            if t.e > 0.0 {
                (toward, rho)
            } else if t.e < 0.0 {
                (rho, toward)
            } else {
                (rho, rho)
            }
        }
    };
    if c >= COEFF_MAX {
        rp = f32::INFINITY;
    }
    if c <= -COEFF_MAX {
        rm = f32::INFINITY;
    }
    (rp, rm)
}

/// The binary v3 cost `min(ρ⁺, ρ⁻)` of a coefficient of value `c` with cover
/// cost `rho` (shadow layers; the direction is chosen at embedding).
pub fn binary_cost(c: i16, rho: f32, si: Option<SiTerm>) -> f32 {
    let (rp, rm) = pm_costs(rho, c, si);
    rp.min(rm)
}

/// `ln(1 + e^{−x})` for `x ≥ 0` (softplus of −x), deterministic.
#[inline]
fn softplus_neg(x: f64) -> f64 {
    if x > 40.0 {
        det_exp(-x)
    } else {
        det_ln(1.0 + det_exp(-x))
    }
}

/// Ternary entropy (nats) of one position at multiplier `lam`, with `a` =
/// cost of the P2-flipping direction and `b` = cost of the other one.
#[inline]
fn ternary_entropy(lam: f64, a: f64, b: f64) -> f64 {
    let ea = if a.is_finite() { det_exp(-lam * a) } else { 0.0 };
    let eb = if b.is_finite() { det_exp(-lam * b) } else { 0.0 };
    let z = 1.0 + ea + eb;
    let ta = if a.is_finite() { ea / z * lam * a } else { 0.0 };
    let tb = if b.is_finite() { eb / z * lam * b } else { 0.0 };
    det_ln(z) + ta + tb
}

/// Solve the Gibbs multiplier λ for a ternary payload of `total` bits over
/// `n` positions, from the subsample `sample` (bisection in log space).
/// Deterministic.
fn solve_lambda(sample: &[(f64, f64)], n: usize, total: usize) -> f64 {
    if sample.is_empty() {
        return 1.0;
    }
    let scale = n as f64 / sample.len() as f64;
    let target = total as f64 * core::f64::consts::LN_2; // nats
    let (mut lo, mut hi) = (-40.0f64, 40.0f64); // ln λ; costs span ~1e-6 .. 1e8
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let lam = det_exp(mid);
        let h: f64 = sample.iter().map(|&(a, b)| ternary_entropy(lam, a, b)).sum::<f64>() * scale;
        if h > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    det_exp(0.5 * (lo + hi))
}

/// Expected distortion of the optimal (Gibbs) ternary embedding of `total`
/// bits over `n` positions, from the subsample `sample` of `(a, b)` cost
/// pairs (pass `b = ∞` for binary). Measurement only.
pub fn bound_distortion(sample: &[(f64, f64)], n: usize, total: usize) -> f64 {
    if sample.is_empty() {
        return 0.0;
    }
    let scale = n as f64 / sample.len() as f64;
    let target = total as f64 * core::f64::consts::LN_2;
    let eval = |lam: f64| -> (f64, f64) {
        let mut h = 0.0;
        let mut d = 0.0;
        for &(a, b) in sample {
            let ea = if a.is_finite() { det_exp(-lam * a) } else { 0.0 };
            let eb = if b.is_finite() { det_exp(-lam * b) } else { 0.0 };
            let z = 1.0 + ea + eb;
            let da = if a.is_finite() { ea / z * a } else { 0.0 };
            let db = if b.is_finite() { eb / z * b } else { 0.0 };
            h += det_ln(z) + lam * (da + db);
            d += da + db;
        }
        (h * scale, d * scale)
    };
    let (mut lo, mut hi) = (-40.0f64, 40.0f64);
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if eval(det_exp(mid)).0 > target { lo = mid } else { hi = mid }
    }
    eval(det_exp(0.5 * (lo + hi))).1
}

// ---------------------------------------------------------------------------
// Lanes
// ---------------------------------------------------------------------------

/// Positions per lane (target): each lane is an independent pair of STCs, so
/// lanes run in parallel and stay on the inline trellis schedule.
pub const LANE_POSITIONS: usize = 1 << 20;
/// Smallest frame share a lane carries (bits), so small messages on large
/// covers use one lane over the whole cover.
pub const LANE_MIN_BITS: usize = 1 << 12;
/// Upper bound on lanes.
pub const MAX_LANES: usize = 64;

/// Lane count for a cover of `n` AC positions and a frame of `m` bits.
pub fn lanes(n: usize, m: usize) -> usize {
    (n / LANE_POSITIONS).min(m / LANE_MIN_BITS).clamp(1, MAX_LANES)
}

/// Positions of a lane when the cover runs at its full lane count (the
/// lane size of a large frame): `n / clamp(⌊n / 2²⁰⌋, 1, 64)`, < 2²¹ for
/// any cover with two or more lanes.
pub fn full_lane_positions(n: usize) -> usize {
    n.div_ceil((n / LANE_POSITIONS).clamp(1, MAX_LANES))
}

/// The in-memory back-pointer table of a full-length lane of this cover
/// (shadow trellises keep at most this much).
pub fn lane_inline_cap(n: usize, h: usize) -> usize {
    trellis::inline_bytes(full_lane_positions(n), h)
}

/// Working set the lanes planned at once may hold together: that of a large
/// frame on this cover — `min(workers, ⌊n / 2²⁰⌋)` full-length lanes, each
/// with its vectors (8 B per position) and its in-memory table.
///
/// Smaller frames get fewer, longer lanes (up to one lane over the whole
/// cover). Sharing this amount between them keeps the peak at or under the
/// large-frame case (measured: a 4 KB message on 12 MP had six 2 M-position
/// lanes with 128 MB of back pointers each — 1.37 GB peak vs 629 MB for a
/// 96 KB message); what a lane's table does not fit is recomputed
/// (`trellis`: tail + checkpointed segments, in waves on idle workers).
/// Every schedule gives the same plan, so the output does not depend on it.
pub fn lanes_in_flight_bytes(n: usize, h: usize, workers: usize) -> usize {
    let nf = full_lane_positions(n);
    let lf = (n / LANE_POSITIONS).clamp(1, MAX_LANES);
    workers.clamp(1, lf).saturating_mul(nf * 8 + trellis::inline_bytes(nf, h))
}

/// Back-pointer bytes each of `k` lanes of `nk` positions planned at once may
/// keep ([`lanes_in_flight_bytes`] shared out, less their vectors).
pub fn lane_inline_share(n: usize, nk: usize, k: usize, h: usize, workers: usize) -> usize {
    (lanes_in_flight_bytes(n, h, workers) / k.max(1)).saturating_sub(nk * 8)
}

/// Workers each of `k` lanes planned at once may use to recompute its
/// trellis (the idle ones, shared out).
pub fn lane_threads(k: usize, workers: usize) -> usize {
    workers.max(1).div_ceil(k.max(1))
}

/// `[k·total/l, (k+1)·total/l)` (u64 arithmetic: portable to 32-bit usize).
pub(crate) fn share(total: usize, l: usize, k: usize) -> std::ops::Range<usize> {
    let a = (k as u64 * total as u64 / l as u64) as usize;
    let b = ((k as u64 + 1) * total as u64 / l as u64) as usize;
    a..b
}

/// Layer split of a lane's `c` frame bits: layer 1 (P1) carries the first
/// `c − c/2`, layer 2 (P2) the last `c/2`. Fixed (wire). The optimal ternary
/// embedding puts a share of 0.504–0.510 of the entropy into P2 at
/// 0.05–0.25 bpnzAC (BOSSbase, Gibbs model), so with a 50/50 split layer 1
/// makes 2–5 % more changes than layer 2 — but the distortion of the fixed
/// split is within 0.02 % of the optimal split's (R3.16 measurement,
/// `ghost-embedding-v3.md`, v3.1). The excess of layer-1 changes was a
/// directional signature only while the pair structure was fixed (v3); with
/// keyed pairs ([`PairKey`]) it has none.
pub fn layer_split(c: usize) -> (usize, usize) {
    (c - c / 2, c / 2)
}

/// Trellis positions a lane of `nk` positions walks for `bits` frame bits
/// (both layers; the unit of the lane progress meter).
fn lane_work(nk: usize, bits: usize) -> usize {
    LaneGeom::new(nk, bits).map_or(0, |g| g.n1() + g.n2())
}

/// Working memory of one lane: per position the flip map (1 B), one layer's
/// cover and cost vectors (5 B, the layers run one after the other) and the
/// trellis plan (1 B) — 8 B with slack — plus the trellis back pointers
/// (in-memory table, or tail / segment buffers + checkpoints).
pub fn lane_bytes(nk: usize, bits: usize, h: usize, inline_limit: usize, threads: usize) -> usize {
    let trellis = match LaneGeom::new(nk, bits) {
        Ok(g) => [g.s1, g.s2]
            .iter()
            .flatten()
            .map(|s| trellis::working_bytes(s.n(), s.avg_w(), h, inline_limit, threads))
            .max()
            .unwrap_or(0),
        Err(_) => trellis::working_bytes(nk, 1, h, inline_limit, threads),
    };
    nk * 8 + trellis
}

/// Encoder knobs that do not change the output.
#[derive(Clone, Debug)]
pub struct EncodeOpts {
    /// STC constraint height ([`V31_STC_H`]; other values only for measurement:
    /// the decoder reads `V31_STC_H`).
    pub h: usize,
    /// Lanes planned at the same time (at most the workers; memory:
    /// [`lane_bytes`] each).
    pub parallel_lanes: usize,
    /// Back-pointer bytes a lane may keep (inline trellis schedule), on top
    /// of the cover's [`lane_inline_share`].
    pub lane_inline_bytes: usize,
    /// Most positions whose costs are held at once: the lanes are planned in
    /// groups of consecutive lanes, each group's costs gathered by one strip
    /// pass over the image (8 B per position while gathering, 4 B after).
    pub group_positions: usize,
    /// Progress steps the lanes report (spread over all trellis work).
    pub progress_steps: u32,
    /// Also estimate the rate-distortion bound (measurement).
    pub bound: bool,
}

impl Default for EncodeOpts {
    fn default() -> Self {
        Self {
            h: V31_STC_H,
            parallel_lanes: usize::MAX,
            lane_inline_bytes: crate::stego::memory::LANE_INLINE_BYTES,
            group_positions: usize::MAX,
            progress_steps: 0,
            bound: false,
        }
    }
}

/// Spreads `steps` progress advances over `total` units of work reported
/// from any thread.
struct Meter {
    total: u64,
    steps: u64,
    done: AtomicU64,
}

impl Meter {
    fn tick(&self, k: usize) {
        if self.steps == 0 || self.total == 0 {
            return;
        }
        let before = self.done.fetch_add(k as u64, Ordering::Relaxed);
        let after = before + k as u64;
        let s0 = (before.min(self.total) * self.steps) / self.total;
        let s1 = (after.min(self.total) * self.steps) / self.total;
        for _ in s0..s1 {
            progress::advance();
        }
    }

    /// Advance whatever is left (rounding, infeasible lanes).
    fn finish(&self) {
        if self.steps == 0 || self.total == 0 {
            return;
        }
        let done = self.done.swap(self.total, Ordering::Relaxed).min(self.total);
        for _ in (done * self.steps) / self.total..self.steps {
            progress::advance();
        }
    }
}

/// Coefficient values of layers planned earlier but not yet written into the
/// image (shadow layers): the planner reads the cover from the image (costs
/// are always the cover's) and these values where they differ.
#[derive(Default, Clone)]
pub struct Overlay {
    map: std::collections::HashMap<u32, i16>,
}

impl Overlay {
    #[inline]
    pub fn get(&self, fi: u32) -> Option<i16> {
        self.map.get(&fi).copied()
    }

    pub fn set(&mut self, fi: u32, v: i16) {
        self.map.insert(fi, v);
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, i16)> + '_ {
        self.map.iter().map(|(&k, &v)| (k, v))
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// The current value of coefficient `fi`: the overlay's, else the cover's.
#[inline]
pub(crate) fn value(grid: &DctGrid, overlay: Option<&Overlay>, fi: u32) -> i16 {
    match overlay.and_then(|o| o.get(fi)) {
        Some(v) => v,
        None => coeff(grid, fi),
    }
}

/// The cover's luminance quantization table.
pub(crate) fn luma_qt(img: &JpegImage) -> Result<&crate::codec::jpeg::dct::QuantTable, StegoError> {
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)
}

/// J-UNIWARD costs of the coefficients `fis` of `cover` (any order): one
/// strip-wise pass, 8 B per requested position.
pub fn gather_costs(cover: &JpegImage, fis: &[u32]) -> Result<Vec<f32>, StegoError> {
    crate::stego::cost::uniward::gather_ac_costs(cover.dct_grid(0), luma_qt(cover)?, fis)
}

/// Work units of one cost pass, in trellis positions (progress meter).
fn gather_work(n: usize) -> u64 {
    (n / 4) as u64
}

/// One lane's planned changes: `(order index within lane, ±1)`.
pub(crate) struct LanePlan {
    changes: Vec<(u32, i16)>,
    distortion: f64,
    w1: usize,
    w2: usize,
    /// Changes made by layer 2 (P2 flips).
    changes2: usize,
}

/// A lane's position costs: aligned with its slice of the order (the cost
/// groups of [`plan`]), or the whole Y channel's by flat index (a plan with
/// one cost group; the split encode, `session.rs`).
#[derive(Clone, Copy)]
pub(crate) enum Rho<'a> {
    Lane(&'a [f32]),
    Flat(&'a [f32]),
}

/// The costs [`plan`] holds for a cost group: aligned with the group's slice
/// of the order, or (a single group covering the cover) the dense costs by
/// flat index.
enum GroupCosts {
    Lane(Vec<f32>),
    Flat(Vec<f32>),
}

/// What a lane reads: the cover grid, the overlay, its slice of the order and
/// its positions' cover costs.
pub(crate) struct LaneInput<'a> {
    pub(crate) grid: &'a DctGrid,
    pub(crate) overlay: Option<&'a Overlay>,
    pub(crate) wet: Option<&'a WetMask>,
    pub(crate) si: Option<&'a SideInfo>,
    pub(crate) ord: &'a [u32],
    pub(crate) rho: Rho<'a>,
    /// The pair bits `s` (by flat index).
    pub(crate) pairs: &'a PairKey,
    /// Every position's value and `(a, b)` up to the cache's length, read
    /// once (the split encode's coordinator; `LaneCache`).
    pub(crate) cache: Option<&'a LaneCache>,
}

/// A lane's values, pair bits and `(a, b)` costs, gathered in one pass: the
/// layer inputs and the lane's changes then read them in order instead of
/// each gathering every position from the grid, the costs and the side
/// information again (random access; on WASM the coordinator's three such
/// passes paced the helpers). 11 B per position. Same values as
/// [`LaneInput`] computes, so the same plan.
pub(crate) struct LaneCache {
    vals: Vec<i16>,
    pairs: Vec<u8>,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl LaneCache {
    /// The first `len` positions of `inp`'s lane.
    pub(crate) fn build(inp: &LaneInput<'_>, len: usize) -> Self {
        let mut vals = Vec::with_capacity(len);
        let mut pairs = Vec::with_capacity(len);
        let mut a = Vec::with_capacity(len);
        let mut b = Vec::with_capacity(len);
        for_each_index(len, |i| {
            vals.push(inp.value(i));
            pairs.push(inp.pair(i) as u8);
            let (x, y) = inp.ab(i);
            a.push(x);
            b.push(y);
        });
        Self { vals, pairs, a, b }
    }
}

/// Indices per call of [`each_chunk`].
const LOOP_CHUNK: usize = 1 << 16;

/// `f(i)` for `i` in `0..n`, `LOOP_CHUNK` indices per call of a function
/// that is not inlined on wasm32: V8 keeps a call in baseline code (Liftoff)
/// until it returns (no on-stack replacement for Wasm), so a lane-long loop
/// in one call (12 M positions) never ran optimized.
#[inline]
fn for_each_index(n: usize, mut f: impl FnMut(usize)) {
    let mut lo = 0;
    while lo < n {
        let hi = (lo + LOOP_CHUNK).min(n);
        each_chunk(lo, hi, &mut f);
        lo = hi;
    }
}

#[cfg_attr(target_arch = "wasm32", inline(never))]
fn each_chunk<F: FnMut(usize)>(lo: usize, hi: usize, f: &mut F) {
    for i in lo..hi {
        f(i);
    }
}

impl LaneInput<'_> {
    #[inline]
    fn value(&self, i: usize) -> i16 {
        if let Some(c) = self.cache
            && i < c.vals.len()
        {
            return c.vals[i];
        }
        value(self.grid, self.overlay, self.ord[i])
    }

    /// The pair bit `s` of position `i`.
    #[inline]
    fn pair(&self, i: usize) -> i16 {
        if let Some(c) = self.cache
            && i < c.pairs.len()
        {
            return c.pairs[i] as i16;
        }
        self.pairs.get(self.ord[i])
    }

    /// `(a, b)`: the cost of the P2-flipping direction, of the other one.
    #[inline]
    fn ab(&self, i: usize) -> (f32, f32) {
        if let Some(c) = self.cache
            && i < c.a.len()
        {
            return (c.a[i], c.b[i]);
        }
        let fi = self.ord[i];
        if self.wet.is_some_and(|w| w.get(fi)) {
            return (f32::INFINITY, f32::INFINITY);
        }
        let c = self.value(i);
        let rho = match self.rho {
            Rho::Lane(r) => r[i],
            Rho::Flat(r) => r[fi as usize],
        };
        let (rp, rm) = pm_costs(rho, c, si_term(self.si, fi as usize));
        if dir2(c, self.pairs.get(fi)) > 0 { (rp, rm) } else { (rm, rp) }
    }
}

/// A lane's layer split and layer shapes.
#[derive(Clone, Copy, Debug)]
pub struct LaneGeom {
    pub(crate) nk: usize,
    pub(crate) m1: usize,
    pub(crate) m2: usize,
    pub(crate) s1: Option<LayerShape>,
    pub(crate) s2: Option<LayerShape>,
}

impl LaneGeom {
    pub fn new(nk: usize, bits: usize) -> Result<Self, StegoError> {
        let (m1, m2) = layer_split(bits);
        let s1 = Some(LayerShape::new(nk, m1).ok_or(StegoError::MessageTooLarge)?);
        let s2 = if m2 > 0 { Some(LayerShape::new(nk, m2).ok_or(StegoError::MessageTooLarge)?) } else { None };
        Ok(Self { nk, m1, m2, s1, s2 })
    }

    /// Positions the layer-1 / layer-2 trellis walks.
    pub(crate) fn n1(&self) -> usize {
        self.s1.map_or(0, |s| s.n())
    }

    pub(crate) fn n2(&self) -> usize {
        self.s2.map_or(0, |s| s.n())
    }

    /// Header (ladder) width of layer 1 / layer 2 (statistics).
    pub(crate) fn w1(&self) -> usize {
        self.s1.map_or(0, |s| s.head_w)
    }

    pub(crate) fn w2(&self) -> usize {
        self.s2.map_or(0, |s| s.head_w)
    }
}

/// Layer 2's trellis input: every P2 flip forces the P2-flipping direction
/// (cost a); binary cost from the marginal flip probability,
/// `a + ln(1 + e^{−λb})/λ`.
pub(crate) fn layer2_input(inp: &LaneInput<'_>, g: &LaneGeom, lam: f64) -> (Vec<u8>, Vec<f32>) {
    let (mut cover2, mut cost2) = (Vec::with_capacity(g.n2()), Vec::with_capacity(g.n2()));
    layer2_input_into(inp, g, lam, &mut cover2, &mut cost2);
    (cover2, cost2)
}

/// [`layer2_input`] into `cover2` / `cost2` (cleared; their allocations
/// reused).
fn layer2_input_into(inp: &LaneInput<'_>, g: &LaneGeom, lam: f64, cover2: &mut Vec<u8>, cost2: &mut Vec<f32>) {
    let n2 = g.n2();
    cover2.clear();
    cost2.clear();
    cover2.reserve(n2);
    cost2.reserve(n2);
    for_each_index(n2, |i| {
        cover2.push(plane2(inp.value(i), inp.pair(i)));
        let (a, b) = inp.ab(i);
        cost2.push(if !a.is_finite() {
            f32::INFINITY
        } else if !b.is_finite() {
            a
        } else {
            (a as f64 + softplus_neg(lam * b as f64) / lam) as f32
        });
    });
}

/// The P2 flips of a planned layer 2 (`nk` entries).
pub(crate) fn layer2_flips(g: &LaneGeom, cover2: &[u8], stego2: &[u8]) -> Vec<bool> {
    let mut flip2 = vec![false; g.nk];
    for (i, (&cb, &sb)) in cover2.iter().zip(stego2).enumerate() {
        flip2[i] = cb != sb;
    }
    flip2
}

/// Layer 1's trellis input: P2-flipped positions are fixed (their P1 flips
/// with them); every other change takes the P2-preserving direction (cost b).
pub(crate) fn layer1_input(inp: &LaneInput<'_>, g: &LaneGeom, flip2: &[bool]) -> (Vec<u8>, Vec<f32>) {
    let (mut cover1, mut cost1) = (Vec::with_capacity(g.n1()), Vec::with_capacity(g.n1()));
    layer1_input_into(inp, g, flip2, &mut cover1, &mut cost1);
    (cover1, cost1)
}

/// [`layer1_input`] into `cover1` / `cost1` (cleared; their allocations
/// reused).
fn layer1_input_into(inp: &LaneInput<'_>, g: &LaneGeom, flip2: &[bool], cover1: &mut Vec<u8>, cost1: &mut Vec<f32>) {
    let n1 = g.n1();
    cover1.clear();
    cost1.clear();
    cover1.reserve(n1);
    cost1.reserve(n1);
    for_each_index(n1, |i| {
        let p = plane1(inp.value(i));
        if flip2[i] {
            cover1.push(p ^ 1);
            cost1.push(f32::INFINITY);
        } else {
            cover1.push(p);
            cost1.push(inp.ab(i).1);
        }
    });
}

/// The lane's changes from its planned layers.
pub(crate) fn lane_plan(inp: &LaneInput<'_>, g: &LaneGeom, flip2: &[bool], cover1: &[u8], stego1: &[u8]) -> LanePlan {
    let n1 = g.n1();
    let mut plan = LanePlan { changes: Vec::new(), distortion: 0.0, w1: g.w1(), w2: g.w2(), changes2: 0 };
    for_each_index(g.nk.min(n1.max(g.n2())), |i| {
        if flip2[i] {
            plan.distortion += inp.ab(i).0 as f64;
            plan.changes.push((i as u32, dir2(inp.value(i), inp.pair(i))));
            plan.changes2 += 1;
        } else if i < n1 && stego1[i] != cover1[i] {
            plan.distortion += inp.ab(i).1 as f64;
            plan.changes.push((i as u32, -dir2(inp.value(i), inp.pair(i))));
        }
    });
    plan
}

/// The code of layer 1 / layer 2 of a lane.
pub fn layer_code(seeds: &V31Seeds, layer: u8, g: &LaneGeom, h: usize) -> Code {
    let (seed, shape) = if layer == 1 { (&seeds.hhat1, g.s1) } else { (&seeds.hhat2, g.s2) };
    shape.expect("the layer exists").code(seed, h)
}

/// Plan one lane: layer 2 (P2) then layer 1 (P1).
/// `inline` / `threads`: the trellis memory schedule (no effect on the plan).
#[allow(clippy::too_many_arguments)]
/// Capacities of a lane's buffers, the same for every lane of a plan
/// ([`LaneCaps::of`]): lanes differ by a few positions, and a freed buffer
/// can serve the next lane only if that one is no larger (an allocator
/// that keeps freed blocks resident — libmalloc on macOS / iOS — otherwise
/// maps a new block and keeps the old one).
#[derive(Clone, Copy, Default)]
pub(crate) struct LaneCaps {
    /// Trellis input (cover bits, costs) of either layer.
    layer: usize,
    /// Positions of the lane (its P2 flips).
    nk: usize,
}

impl LaneCaps {
    /// The largest layer and lane of the `l` lanes splitting `n` positions
    /// and `m` frame bits.
    pub(crate) fn of(n: usize, m: usize, l: usize) -> Self {
        (0..l).fold(Self::default(), |c, k| {
            let nk = share(n, l, k).len();
            let layer = LaneGeom::new(nk, share(m, l, k).len()).map_or(0, |g| g.n1().max(g.n2()));
            Self { layer: c.layer.max(layer), nk: c.nk.max(nk) }
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn plan_lane(
    inp: &LaneInput<'_>,
    bits: &[u8],
    lam: f64,
    seeds: &V31Seeds,
    opts: &EncodeOpts,
    inline: usize,
    threads: usize,
    caps: LaneCaps,
    meter: &Meter,
) -> Result<LanePlan, StegoError> {
    let g = LaneGeom::new(inp.ord.len(), bits.len())?;
    let h = opts.h;
    let (l1, l2) = bits.split_at(g.m1);
    let tick = |k: usize| meter.tick(k);

    // One trellis input buffer pair for both layers.
    let cap = caps.layer.max(g.n1()).max(g.n2());
    let (mut cover, mut cost) = (Vec::with_capacity(cap), Vec::with_capacity(cap));
    let r2 = if g.m2 > 0 {
        layer2_input_into(inp, &g, lam, &mut cover, &mut cost);
        let code2 = layer_code(seeds, 2, &g, h);
        Some(trellis::embed_code(&cover, &cost, l2, &code2, h, inline, threads, &tick)?.ok_or(StegoError::MessageTooLarge)?)
    } else {
        None
    };
    let mut flip2 = Vec::with_capacity(caps.nk.max(g.nk));
    flip2.resize(g.nk, false);
    if let Some(r2) = r2 {
        for (f, (&cb, &sb)) in flip2.iter_mut().zip(cover.iter().zip(&r2.stego_bits)) {
            *f = cb != sb;
        }
    }
    layer1_input_into(inp, &g, &flip2, &mut cover, &mut cost);
    let code1 = layer_code(seeds, 1, &g, h);
    let r1 = trellis::embed_code(&cover, &cost, l1, &code1, h, inline, threads, &tick)?
        .ok_or(StegoError::MessageTooLarge)?;
    Ok(lane_plan(inp, &g, &flip2, &cover, &r1.stego_bits))
}

/// Order positions whose costs feed the median (first `n_med`) and the λ
/// solve (the rest).
pub(crate) fn sample_positions(n: usize) -> (Vec<usize>, usize) {
    let med_stride = (n / 65536).max(1);
    let lam_stride = (n / LAMBDA_SAMPLES).max(1);
    let n_med = n.div_ceil(med_stride);
    ((0..n).step_by(med_stride).chain((0..n).step_by(lam_stride)).collect(), n_med)
}

/// λ (layer-2 cost), median cost and the λ sample's `(a, b)` pairs from the
/// cover costs of [`sample_positions`] (`sample_rho`, aligned).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_stats(
    ord: &[u32],
    samples: &[usize],
    n_med: usize,
    sample_rho: &[f32],
    m: usize,
    grid: &DctGrid,
    overlay: Option<&Overlay>,
    wet: Option<&WetMask>,
    si: Option<&SideInfo>,
    pairs: &PairKey,
) -> (f64, f32, Vec<(f64, f64)>) {
    let n = ord.len();
    // Sample costs → (a, b) pairs exactly as the lanes form them.
    let ab_at = |k: usize, rho: f32| -> (f32, f32) {
        let fi = ord[k];
        if wet.is_some_and(|w| w.get(fi)) {
            return (f32::INFINITY, f32::INFINITY);
        }
        let c = value(grid, overlay, fi);
        let (rp, rm) = pm_costs(rho, c, si_term(si, fi as usize));
        if dir2(c, pairs.get(fi)) > 0 { (rp, rm) } else { (rm, rp) }
    };
    let mut median_cost = 0.0;
    let mut med: Vec<f32> = samples[..n_med]
        .iter()
        .zip(&sample_rho[..n_med])
        .map(|(&k, &rho)| {
            let (a, b) = ab_at(k, rho);
            a.min(b)
        })
        .filter(|c| c.is_finite())
        .collect();
    if !med.is_empty() {
        let mid = med.len() / 2;
        med.select_nth_unstable_by(mid, f32::total_cmp);
        median_cost = med[mid];
    }
    let sample_ab: Vec<(f64, f64)> = samples[n_med..]
        .iter()
        .zip(&sample_rho[n_med..])
        .map(|(&k, &rho)| {
            let (a, b) = ab_at(k, rho);
            (a as f64, b as f64)
        })
        .collect();
    let lam = solve_lambda(&sample_ab, n, m);
    (lam, median_cost, sample_ab)
}

/// The plan of all lanes (in lane order) as coefficient changes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble(
    plans: Vec<LanePlan>,
    ord: &[u32],
    grid: &DctGrid,
    overlay: Option<&Overlay>,
    m: usize,
    median_cost: f32,
    sample_ab: &[(f64, f64)],
    bound: bool,
) -> V31Plan {
    let n = ord.len();
    let l = plans.len();
    let mut stats = V31Stats { n, m, median_cost, lanes: l, ..Default::default() };
    if bound {
        stats.bound_distortion = bound_distortion(sample_ab, n, m);
    }
    let mut changes = Vec::with_capacity(plans.iter().map(|p| p.changes.len()).sum());
    for (k, p) in plans.into_iter().enumerate() {
        if k == 0 {
            stats.w1 = p.w1;
            stats.w2 = p.w2;
            stats.m2 = layer_split(share(m, l, 0).len()).1;
        }
        let base = share(n, l, k).start;
        for (i, d) in p.changes {
            let fi = ord[base + i as usize];
            changes.push((fi, value(grid, overlay, fi) + d));
        }
        stats.changes += changes.len() - stats.changes;
        stats.changes2 += p.changes2;
        stats.distortion += p.distortion;
    }
    V31Plan { stats, changes }
}

/// A planned embedding: statistics and the new value of every changed
/// coefficient (`(flat index, value)`, values relative to the overlay).
pub struct V31Plan {
    pub stats: V31Stats,
    pub changes: Vec<(u32, i16)>,
}

/// Plan the embedding of `frame_bits` into the Y channel of `cover` (read
/// only; costs are always the cover's).
///
/// `wet`: positions owned by shadow layers and the dummy blocks (never
/// changed; the primary's order skips nothing, their bits are simply not the
/// primary's to set). Build it with [`wet_mask_for`]; `None` = the dummy
/// blocks only.
/// `overlay`: values of those layers not yet written into `cover`.
/// The plan is independent of `opts.parallel_lanes`, `opts.lane_inline_bytes`,
/// `opts.group_positions` and `opts.progress_steps`.
pub fn plan(
    cover: &JpegImage,
    frame_bits: &[u8],
    structural_key: &[u8; 64],
    si: Option<&SideInfo>,
    wet: Option<&WetMask>,
    overlay: Option<&Overlay>,
    opts: &EncodeOpts,
) -> Result<V31Plan, StegoError> {
    let t = tstart();
    // Callers that pass a mask build it with `wet_mask_for` (dummy blocks
    // included); without one, the dummy blocks are the only wet positions.
    let dummies = if wet.is_none() { dummy_block_wet(cover) } else { None };
    let wet = wet.or(dummies.as_ref());
    let seeds = v31_seeds(structural_key);
    let grid = cover.dct_grid(0);
    let ord = order(grid, &seeds);
    tlog("order", t);
    let pairs = PairKey::new(&seeds.pair, grid.coeffs().len());
    let n = ord.len();
    let m = frame_bits.len();
    if m == 0 || m > n {
        return Err(StegoError::MessageTooLarge);
    }
    progress::check_cancelled()?;

    let l = lanes(n, m);
    let caps = LaneCaps::of(n, m, l);
    let workers = crate::stego::memory::encode_workers();

    // Groups of consecutive lanes whose positions fit `group_positions`.
    let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
    let mut k0 = 0;
    while k0 < l {
        let mut k1 = k0 + 1;
        while k1 < l && share(n, l, k1).end - share(n, l, k0).start <= opts.group_positions {
            k1 += 1;
        }
        groups.push(k0..k1);
        k0 = k1;
    }
    let meter = Meter {
        total: (0..l).map(|k| lane_work(share(n, l, k).len(), share(m, l, k).len()) as u64).sum::<u64>()
            + groups.len() as u64 * gather_work(n),
        steps: opts.progress_steps as u64,
        done: AtomicU64::new(0),
    };

    // The cost samples (median for the quality score, λ for the payload
    // split) are gathered with the first group.
    let (samples, n_med) = sample_positions(n);

    let mut lam = 1.0;
    let mut median_cost = 0.0;
    let mut sample_ab: Vec<(f64, f64)> = Vec::new();
    let mut plans: Vec<LanePlan> = Vec::with_capacity(l);
    for (g, ks) in groups.iter().enumerate() {
        let r = share(n, l, ks.start).start..share(n, l, ks.end - 1).end;
        let rho = if groups.len() == 1 {
            // One group (the whole cover): the lanes read the image's dense
            // costs by flat index. Gathering them in the order's sequence
            // instead was a second array of the same size, freed before the
            // lanes: the allocator keeps such a block resident (macOS / iOS
            // large-allocation cache) and the lane tables, larger, cannot
            // reuse it — +46 MB of footprint at 12 MP for no live data.
            // Same values as the gather, so the same plan.
            let dense = crate::stego::cost::uniward::dense_ac_costs(grid, luma_qt(cover)?)?;
            let tail: Vec<f32> = samples.iter().map(|&k| dense[ord[k] as usize]).collect();
            (lam, median_cost, sample_ab) = sample_stats(&ord, &samples, n_med, &tail, m, grid, overlay, wet, si, &pairs);
            tlog("lambda", t);
            GroupCosts::Flat(dense)
        } else if g == 0 {
            // The group's positions and the samples in one cost pass, without
            // copying the group's order (a copy grown by the samples was 8 B
            // per position on top of the gather's 8).
            let sample_fis: Vec<u32> = samples.iter().map(|&k| ord[k]).collect();
            let mut all = crate::stego::cost::uniward::gather_ac_costs_parts(
                grid, luma_qt(cover)?, &ord[r.clone()], &sample_fis,
            )?;
            drop(sample_fis);
            let tail = all.split_off(r.len());
            (lam, median_cost, sample_ab) = sample_stats(&ord, &samples, n_med, &tail, m, grid, overlay, wet, si, &pairs);
            tlog("lambda", t);
            GroupCosts::Lane(all)
        } else {
            GroupCosts::Lane(gather_costs(cover, &ord[r.clone()])?)
        };
        meter.tick(gather_work(n) as usize);

        // A batch of `nb` lanes shares the in-flight working set; idle
        // workers help recompute the trellises of a short batch.
        let plan_k = |k: usize, nb: usize| -> Result<LanePlan, StegoError> {
            let lr = share(n, l, k);
            let inline = opts.lane_inline_bytes.min(lane_inline_share(n, lr.len(), nb, opts.h, workers));
            let inp = LaneInput {
                grid,
                overlay,
                wet,
                si,
                ord: &ord[lr.clone()],
                rho: match &rho {
                    GroupCosts::Lane(v) => Rho::Lane(&v[lr.start - r.start..lr.end - r.start]),
                    GroupCosts::Flat(d) => Rho::Flat(d),
                },
                pairs: &pairs,
                cache: None,
            };
            let threads = lane_threads(nb, workers);
            plan_lane(&inp, &frame_bits[share(m, l, k)], lam, &seeds, opts, inline, threads, caps, &meter)
        };
        let batch = opts.parallel_lanes.min(workers).clamp(1, ks.len());
        for b0 in (ks.start..ks.end).step_by(batch) {
            let kb = b0..(b0 + batch).min(ks.end);
            let nb = kb.len();
            #[cfg(feature = "parallel")]
            let res: Vec<Result<LanePlan, StegoError>> = {
                use rayon::prelude::*;
                kb.into_par_iter().map(|k| plan_k(k, nb)).collect()
            };
            #[cfg(not(feature = "parallel"))]
            let res: Vec<Result<LanePlan, StegoError>> = kb.map(|k| plan_k(k, nb)).collect();
            for p in res {
                plans.push(p?);
            }
        }
    }
    meter.finish();
    tlog("lanes", t);
    Ok(assemble(plans, &ord, grid, overlay, m, median_cost, &sample_ab, opts.bound))
}

/// Embed `frame_bits` into `img`'s Y channel (no shadow layers): [`plan`]
/// against the image, then write the changes.
pub fn encode(
    img: &mut JpegImage,
    frame_bits: &[u8],
    structural_key: &[u8; 64],
    si: Option<&SideInfo>,
    opts: &EncodeOpts,
) -> Result<V31Stats, StegoError> {
    let p = plan(img, frame_bits, structural_key, si, None, None, opts)?;
    let coeffs = img.dct_grid_mut(0).coeffs_mut();
    for &(fi, v) in &p.changes {
        coeffs[fi as usize] = v;
    }
    Ok(p.stats)
}

/// The v3 direction rule for a single ±1 change chosen by a binary code
/// (shadow layers): toward the precover with side information, else a keyed
/// coin; never past ±1023.
pub fn modify_pm1(c: i16, si_err: Option<f32>, seed: u64, fi: u64) -> i16 {
    if c >= COEFF_MAX {
        return c - 1;
    }
    if c <= -COEFF_MAX {
        return c + 1;
    }
    let up = match si_err {
        Some(e) if e > 0.0 => true,
        Some(e) if e < 0.0 => false,
        _ => splitmix(fi ^ seed) & 1 == 1,
    };
    if up { c + 1 } else { c - 1 }
}

/// Shadow STC widths over an all-AC set of `n` positions: the ladder from 2
/// while a minimal shadow frame (`min_bits`) still fits.
pub fn shadow_widths(n: usize, min_bits: usize) -> Vec<usize> {
    ladder()
        .skip_while(|&w| w < 2)
        .take_while(|&w| w.saturating_mul(min_bits.max(1)) <= n)
        .collect()
}

/// Phase timing for measurement (`PHASM_GHOST_V3_TIMING`); never on wasm32,
/// where `Instant::now` is unavailable.
pub(crate) fn tstart() -> Option<std::time::Instant> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if std::env::var_os("PHASM_GHOST_V3_TIMING").is_some() {
            return Some(std::time::Instant::now());
        }
    }
    None
}

pub(crate) fn tlog(what: &str, t: Option<std::time::Instant>) {
    if let Some(t) = t {
        eprintln!("v3time {what} {:.3}s", t.elapsed().as_secs_f64());
    }
}

#[inline]
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Direction seed of a shadow layer's coin, from its frame's salt.
pub fn dir_seed_from_salt(salt: &[u8]) -> u64 {
    let h = label_hash(b"phasm-ghost-v3/dir\0\0", salt);
    u64::from_le_bytes(h[..8].try_into().unwrap())
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

fn bits_to_usize(bits: &[u8]) -> usize {
    bits.iter().fold(0usize, |acc, &b| (acc << 1) | (b & 1) as usize)
}

/// Frame bit count announced by a frame prefix (`≥ 48` bits), if plausible.
fn frame_bits_from_prefix(fb: &[u8]) -> Option<usize> {
    let v1 = bits_to_usize(&fb[..16]);
    if v1 > 0 {
        return Some((frame::FRAME_OVERHEAD + v1) * 8);
    }
    let v2 = bits_to_usize(&fb[16..48]);
    if v2 > u16::MAX as usize {
        return Some((frame::FRAME_OVERHEAD_EXT + v2) * 8);
    }
    None
}

/// The first `k` syndrome bits of `code` over the bit-packed positions from
/// `start` of `words`.
fn extract_code(words: &[u64], start: usize, code: &Code, k: usize) -> Vec<u8> {
    trellis::syndrome_prefix(code, k, |j| {
        let j = start + j;
        (words[j >> 6] >> (j & 63)) & 1 != 0
    })
}

/// Smallest frame the encoder writes (bits): overhead + 1 byte.
const MIN_FRAME_BITS: usize = (frame::FRAME_OVERHEAD + 1) * 8;

/// The frame sizes `m` (bits, `MIN_FRAME_BITS..=n`) that give exactly `l`
/// lanes on a cover of `n` positions — an interval, since [`lanes`] is
/// non-decreasing in `m`.
fn lane_m_range(n: usize, l: usize) -> Option<(usize, usize)> {
    if n < MIN_FRAME_BITS {
        return None;
    }
    // First m with lanes(n, m) >= target (n + 1 if none).
    let first_at_least = |target: usize| {
        let (mut lo, mut hi) = (MIN_FRAME_BITS, n + 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if lanes(n, mid) >= target { hi = mid } else { lo = mid + 1 }
        }
        lo
    };
    let lo = first_at_least(l);
    let hi = first_at_least(l + 1) - 1;
    (lo <= hi && lo <= n && lanes(n, lo) == l).then_some((lo, hi.min(n)))
}

/// The bit planes a Ghost decode reads along the keyed order (shared by v3.1
/// and v3), one bit per position.
pub struct DecodePlanes {
    pub n: usize,
    /// `c mod 2` (both generations).
    pub p1: Vec<u64>,
    /// v3.1's `⌊(c + s)/2⌋ mod 2`.
    pub p2: Vec<u64>,
    /// v3's `⌊c/2⌋ mod 2` (`v3_legacy`).
    pub p2_v3: Vec<u64>,
}

/// The planes of `img` along the keyed order of `structural_key`: one
/// shuffle (the order itself, 4 B per position, is dropped at once).
pub fn decode_planes(img: &JpegImage, structural_key: &[u8; 64]) -> DecodePlanes {
    let seeds = v31_seeds(structural_key);
    // The order is v3's (shared on purpose, see the module docs).
    debug_assert_eq!(seeds.perm, super::v3_legacy::perm_seed(structural_key));
    let grid = img.dct_grid(0);
    let ord = order(grid, &seeds);
    let pairs = PairKey::new(&seeds.pair, grid.coeffs().len());
    let n = ord.len();
    let mut p1 = vec![0u64; n.div_ceil(64)];
    let mut p2 = vec![0u64; n.div_ceil(64)];
    let mut p2_v3 = vec![0u64; n.div_ceil(64)];
    for (i, &fi) in ord.iter().enumerate() {
        let c = coeff(grid, fi);
        p1[i >> 6] |= (plane1(c) as u64) << (i & 63);
        p2[i >> 6] |= (plane2(c, pairs.get(fi)) as u64) << (i & 63);
        p2_v3[i >> 6] |= (super::v3_legacy::plane2(c) as u64) << (i & 63);
    }
    DecodePlanes { n, p1, p2, p2_v3 }
}

/// Try to decode a v3.1 primary from its planes. `None` = no v3.1 frame
/// found for this passphrase; `Some(Err(DecryptionFailed))` = a v3.1 frame
/// with a valid CRC whose ciphertext did not authenticate.
///
/// Search: for each lane count `l` the cover admits, and each ladder width
/// `w` lane 0's layer-1 header can have with `l` lanes (from the interval of
/// frame sizes that give `l` lanes), read lane 0's first 48 layer-1 syndrome
/// bits (the frame header, in the code's 48 header blocks of width `w`) and
/// accept the candidate only if the announced frame length gives back
/// exactly `l` lanes and header width `w`. Everything else is then
/// determined.
pub fn decode_v31(
    planes: &DecodePlanes,
    passphrase: &str,
    structural_key: &[u8; 64],
    parse_and_decrypt: &dyn Fn(&[u8], &str) -> Result<PayloadData, StegoError>,
) -> Option<Result<PayloadData, StegoError>> {
    let n = planes.n;
    if n == 0 {
        return None;
    }
    let t = tstart();
    let seeds = v31_seeds(structural_key);
    let h = V31_STC_H;
    let (p1, p2) = (&planes.p1, &planes.p2);
    let mut saw_decrypt_fail = false;
    let l_max = (n / LANE_POSITIONS).clamp(1, MAX_LANES);
    for l in 1..=l_max {
        let Some((m_lo, m_hi)) = lane_m_range(n, l) else { continue };
        let r0 = share(n, l, 0);
        let n0 = r0.len();
        let m1_of = |m: usize| layer_split(share(m, l, 0).len()).0;
        let (Some(w_max), w_min) = (choose_w(n0, m1_of(m_lo)), choose_w(n0, m1_of(m_hi)).unwrap_or(1)) else {
            continue;
        };
        for w in ladder().skip_while(|&w| w < w_min).take_while(|&w| w <= w_max) {
            if n0 / w < HEADER_BLOCKS {
                break;
            }
            let head = Code::uniform(&packed_cols(&v31_hhat(w, &seeds.hhat1, h), w), HEADER_BLOCKS);
            let pre = extract_code(p1, r0.start, &head, HEADER_BLOCKS);
            let Some(m) = frame_bits_from_prefix(&pre) else { continue };
            if m > n || lanes(n, m) != l {
                continue;
            }
            let (m1_0, _) = layer_split(share(m, l, 0).len());
            if LayerShape::new(n0, m1_0).map(|s| (s.head, s.head_w)) != Some((HEADER_BLOCKS, w)) {
                continue;
            }
            // Candidate accepted: read every lane.
            let mut fbits: Vec<u8> = Vec::with_capacity(m);
            let mut ok = true;
            for k in 0..l {
                let rk = share(n, l, k);
                let Ok(g) = LaneGeom::new(rk.len(), share(m, l, k).len()) else {
                    ok = false;
                    break;
                };
                fbits.extend(extract_code(p1, rk.start, &layer_code(&seeds, 1, &g, h), g.m1));
                if g.m2 > 0 {
                    fbits.extend(extract_code(p2, rk.start, &layer_code(&seeds, 2, &g, h), g.m2));
                }
            }
            if !ok {
                continue;
            }
            let frame_bytes = frame::bits_to_bytes(&fbits);
            tlog("dec candidate extracted", t);
            match parse_and_decrypt(&frame_bytes, passphrase) {
                Ok(p) => return Some(Ok(p)),
                Err(StegoError::DecryptionFailed) => saw_decrypt_fail = true,
                Err(_) => {}
            }
        }
    }
    tlog("dec search done", t);
    if saw_decrypt_fail { Some(Err(StegoError::DecryptionFailed)) } else { None }
}

/// Decode a Ghost primary of the generations that share the keyed order —
/// v3.1, then v3 ([`super::v3_legacy`], decode-only) — from one shuffle.
/// `None` = neither found a frame for this passphrase (the caller then tries
/// v1); `Some(Err(DecryptionFailed))` = a frame with a valid CRC whose
/// ciphertext did not authenticate (and no success).
pub fn decode_all(
    img: &JpegImage,
    passphrase: &str,
    structural_key: &[u8; 64],
    parse_and_decrypt: &dyn Fn(&[u8], &str) -> Result<PayloadData, StegoError>,
) -> Option<Result<PayloadData, StegoError>> {
    if img.num_components() == 0 {
        return None;
    }
    let t = tstart();
    let planes = decode_planes(img, structural_key);
    tlog("dec order+planes", t);
    let mut fail = None;
    match decode_v31(&planes, passphrase, structural_key, parse_and_decrypt) {
        Some(Ok(p)) => return Some(Ok(p)),
        Some(Err(e)) => fail = Some(Err(e)),
        None => {}
    }
    match super::v3_legacy::decode_planes(&planes.p1, &planes.p2_v3, planes.n, passphrase, structural_key, parse_and_decrypt) {
        Some(Ok(p)) => Some(Ok(p)),
        Some(Err(e)) => Some(Err(e)),
        None => fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stego::stc::{embed, extract};

    /// `bound_distortion` over a strided subsample of all positions.
    fn bound_of(ab: &[(f32, f32)], n: usize, m: usize) -> f64 {
        let stride = (ab.len() / LAMBDA_SAMPLES).max(1);
        let sample: Vec<(f64, f64)> = ab.iter().step_by(stride).map(|&(a, b)| (a as f64, b as f64)).collect();
        bound_distortion(&sample, n, m)
    }

    /// A smooth 4:2:0 cover with an odd luma block count both ways
    /// (648x488: 81x61 real blocks in an 82x62 MCU grid): on such content the
    /// dummy blocks' costs equal the real ones, and before 2026-11-02 v3 wrote
    /// non-zero AC into them (fix list 2.10). Every encode path, at the bands'
    /// maximum, must leave them exactly as the cover had them.
    #[test]
    fn dummy_blocks_untouched_at_max_payload() {
        use crate::stego::payload::compressed_payload_size;
        use crate::stego::{deep_cover_jpeg, ghost_bands, ghost_bands_si, ShadowLayer};
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        let (w, h) = (648u32, 488u32);
        let mut px = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                // A soft diagonal gradient with a faint ripple (no texture).
                let r = (x * 3 / 4 + y / 3) as f32 + 2.0 * ((x as f32) / 37.0).sin();
                px.push(r.clamp(0.0, 255.0) as u8);
                px.push((40 + x / 6 + y * 3 / 8) as u8);
                px.push((200 - y / 4) as u8);
            }
        }
        let cover = deep_cover_jpeg(&px, w, h, 92).unwrap();
        let cimg = JpegImage::from_bytes(&cover).unwrap();
        let dummies = dummy_block_wet(&cimg).expect("odd block counts give dummy blocks");
        assert_eq!(real_luma_blocks(&cimg), (81, 61));
        let n = cimg.dct_grid(0).coeffs().len();
        let dummy_pos: Vec<usize> = (0..n).filter(|&i| dummies.get(i as u32)).collect();
        assert_eq!(dummy_pos.len(), (82 * 62 - 81 * 61) * 64);

        // The longest random text whose compressed payload fits `max`.
        let msg_for = |max: usize| -> String {
            let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(max as u64);
            let all: String = (0..max * 2).map(|_| (b'a' + rng.gen_range08(0..26u8)) as char).collect();
            let (mut lo, mut hi) = (0usize, all.len());
            while lo < hi {
                let mid = (lo + hi).div_ceil(2);
                if compressed_payload_size(&all[..mid], &[]) <= max { lo = mid } else { hi = mid - 1 }
            }
            all[..lo].to_string()
        };
        let check = |what: &str, stego: &[u8], pass: &str, msg: &str| {
            let simg = JpegImage::from_bytes(stego).unwrap();
            let (c, s) = (cimg.dct_grid(0).coeffs(), simg.dct_grid(0).coeffs());
            let changed = dummy_pos.iter().filter(|&&i| c[i] != s[i]).count();
            assert_eq!(changed, 0, "{what}: {changed} dummy-block coefficients changed");
            assert_eq!(crate::stego::ghost_decode(stego, pass).unwrap().text, msg, "{what}");
        };
        let _seed = crate::stego::crypto::DeterministicSeedGuard::set("2102");

        let plain_msg = msg_for(ghost_bands(&cimg, 0, 0).unwrap().max_bytes);
        check("plain", &crate::stego::ghost_encode(&cover, &plain_msg, "pw").unwrap(), "pw", &plain_msg);

        let si_msg = msg_for(ghost_bands_si(&cimg, &px, w, h, 0, 0).unwrap().max_bytes);
        check("deep cover", &crate::stego::ghost_encode_si(&cover, &px, w, h, &si_msg, "pw").unwrap(), "pw", &si_msg);

        let (split, _, _) =
            crate::stego::run_split_encode_locally(&cover, &si_msg, &[], "pw", Some((&px, w, h)), 3, None).unwrap();
        check("split session", &split, "pw", &si_msg);

        let shadow = ShadowLayer { message: "shadow".into(), passphrase: "other".into(), files: vec![] };
        let sh_msg = msg_for(ghost_bands(&cimg, 1, 6).unwrap().max_bytes);
        let with = crate::stego::ghost_encode_with_shadows(&cover, &sh_msg, &[], "pw", &[shadow], None).unwrap();
        check("shadows", &with, "pw", &sh_msg);
        assert_eq!(crate::stego::ghost_shadow_decode(&with, "other").unwrap().text, "shadow");
    }

    #[test]
    fn ladder_is_increasing_and_dense_at_start() {
        let l: Vec<usize> = ladder().take(40).collect();
        assert_eq!(&l[..16], &(1..=16).collect::<Vec<_>>()[..]);
        assert!(l.windows(2).all(|p| p[1] > p[0]));
        assert_eq!(l[16], 19);
    }

    #[test]
    fn choose_w_is_widest_fitting() {
        assert_eq!(choose_w(100, 0), None);
        assert_eq!(choose_w(100, 101), None);
        assert_eq!(choose_w(100, 100), Some(1));
        assert_eq!(choose_w(100, 10), Some(10));
        assert_eq!(choose_w(1000, 50), Some(19)); // 20 is not on the ladder
        for n in [1000usize, 77_777, 12_000_000] {
            for m in [1usize, 7, 400, 4096, 99_999] {
                if let Some(w) = choose_w(n, m) {
                    assert!(m * w <= n);
                    let next = ladder().find(|&x| x > w).unwrap();
                    assert!(m * next > n);
                }
            }
        }
    }

    #[test]
    fn planes_and_direction() {
        for s in 0..2i16 {
            for c in (-1023i16..=1023).filter(|c| c.abs() <= 50 || c.abs() >= 1020) {
                let up = c + 1;
                let dn = c - 1;
                assert_ne!(plane1(c), plane1(up));
                assert_ne!(plane1(c), plane1(dn));
                let flips_up = plane2(c, s) != plane2(up, s);
                let flips_dn = plane2(c, s) != plane2(dn, s);
                assert!(flips_up ^ flips_dn, "c={c} s={s}");
                assert_eq!(dir2(c, s) > 0, flips_up, "c={c} s={s}");
            }
        }
        // s = 0 is v3's plane 2; s = 1 shifts the pairs by one.
        for c in -50i16..=50 {
            assert_eq!(plane2(c, 0), crate::stego::ghost::v3_legacy::plane2(c));
            assert_eq!(plane2(c, 1), crate::stego::ghost::v3_legacy::plane2(c + 1));
        }
    }

    /// Crafted JPEGs can carry |c| up to 32767 (magnitude category 15): the
    /// pair bits must not overflow and must match the exact i32 arithmetic.
    #[test]
    fn plane2_extreme_coefficients() {
        for s in 0..2i16 {
            for c in [i16::MIN, i16::MIN + 1, -32766, 32766, i16::MAX] {
                let exact = (((c as i32 + s as i32) >> 1) & 1) as u8;
                assert_eq!(plane2(c, s), exact, "c={c} s={s}");
                let odd = (c as i32 + s as i32) & 1 != 0;
                assert_eq!(dir2(c, s) > 0, odd, "c={c} s={s}");
            }
        }
    }

    /// The keyed pair bits are balanced and differ between keys.
    #[test]
    fn pair_key_is_balanced() {
        let a = PairKey::new(&[1u8; 32], 1 << 20);
        let b = PairKey::new(&[2u8; 32], 1 << 20);
        let ones = (0..1u32 << 20).filter(|&fi| a.get(fi) == 1).count();
        assert!((ones as f64 / (1 << 20) as f64 - 0.5).abs() < 0.005, "{ones}");
        let same = (0..1u32 << 20).filter(|&fi| a.get(fi) == b.get(fi)).count();
        assert!((same as f64 / (1 << 20) as f64 - 0.5).abs() < 0.005, "{same}");
    }

    /// v3.1 reads its order from v3's seed (one shuffle decodes both).
    #[test]
    fn order_is_shared_with_v3() {
        let key = [0x5au8; 64];
        assert_eq!(v31_seeds(&key).perm, crate::stego::ghost::v3_legacy::perm_seed(&key));
    }

    /// Diagnostic: STC distortion vs the binary bound on synthetic profiles.
    /// `cargo test --release stc_coding_loss -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn stc_coding_loss() {
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
        let n = 200_000usize;
        for profile in ["constant", "uniform", "lognormal"] {
            for inv_alpha in [2usize, 4, 10, 20, 50, 100] {
                let m = n / inv_alpha;
                let costs: Vec<f32> = (0..n)
                    .map(|i| match profile {
                        "uniform" => rng.gen08::<f32>() + 1e-3,
                        "constant" => 1.0,
                        "square" => {
                            let x = (i as f32 + 1.0) / n as f32;
                            x * x
                        }
                        _ => (rng.gen08::<f32>() * 8.0).exp(),
                    })
                    .collect();
                let cover: Vec<u8> = (0..n).map(|_| rng.gen08::<u8>() & 1).collect();
                let msg: Vec<u8> = (0..m).map(|_| rng.gen08::<u8>() & 1).collect();
                let w = n / m;
                let hh = hhat::generate_hhat(V31_STC_H, w, &[3u8; 32]);
                let r = embed::stc_embed(&cover, &costs, &msg, &hh, V31_STC_H, w).unwrap();
                let hh_e = hhat::generate_hhat_end_set(V31_STC_H, w, &[3u8; 32]);
                let re = embed::stc_embed(&cover, &costs, &msg, &hh_e, V31_STC_H, w).unwrap();
                assert_eq!(extract::stc_extract_prefix(&r.stego_bits, &hh, w, m), msg);
                let ab: Vec<(f32, f32)> = costs.iter().map(|&c| (c, f32::INFINITY)).collect();
                let b = bound_of(&ab, n, m);
                eprintln!("{profile:9} 1/alpha={inv_alpha:3} w={w:3} stc={:.1} bound={:.1} loss={:.3} loss_endset={:.3} changes={}", r.total_cost, b, r.total_cost / b, re.total_cost / b, r.num_modifications);
            }
        }
    }

    /// Diagnostic: spread of the coding loss over random end-set H-hat
    /// matrices (`cargo test --release hhat_search -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn hhat_search() {
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        let n = 60_000usize;
        let ws: Vec<usize> = std::env::var("HHAT_WS")
            .ok()
            .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
            .unwrap_or(vec![2, 3, 4, 5, 8, 10, 16, 26, 52]);
        let tries: usize = std::env::var("HHAT_TRIES").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        // Fixed workloads: 3 uniform-cost covers + messages.
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(99);
        let loads: Vec<(Vec<u8>, Vec<f32>, Vec<u8>)> = (0..3)
            .map(|_| {
                let costs: Vec<f32> = (0..n).map(|_| rng.gen08::<f32>() + 1e-3).collect();
                let cover: Vec<u8> = (0..n).map(|_| rng.gen08::<u8>() & 1).collect();
                let msg: Vec<u8> = (0..n).map(|_| rng.gen08::<u8>() & 1).collect();
                (cover, costs, msg)
            })
            .collect();
        for &w in &ws {
            let m = n / w;
            let bound: f64 = loads
                .iter()
                .map(|(_, c, _)| {
                    let ab: Vec<(f32, f32)> = c.iter().map(|&x| (x, f32::INFINITY)).collect();
                    bound_of(&ab, n, m)
                })
                .sum();
            let mut losses: Vec<(f64, Vec<u32>)> = Vec::new();
            let mut srng = rand_chacha::ChaCha20Rng::seed_from_u64(w as u64);
            for t in 0..tries {
                let cols: Vec<u32> = if t == 0 {
                    // the keyed end-set construction, seed [3; 32]
                    let hh = hhat::generate_hhat_end_set(V31_STC_H, w, &[3u8; 32]);
                    (0..w).map(|c| hhat::column_packed(&hh, c)).collect()
                } else {
                    (0..w).map(|_| (srng.gen08::<u32>() & 0x7f) | 0x41).collect()
                };
                let mut hh = vec![vec![0u32; w]; V31_STC_H];
                for (c, &v) in cols.iter().enumerate() {
                    for r in 0..V31_STC_H {
                        hh[r][c] = (v >> r) & 1;
                    }
                }
                let d: f64 = loads
                    .iter()
                    .map(|(cv, c, msg)| embed::stc_embed(cv, c, &msg[..m], &hh, V31_STC_H, w).unwrap().total_cost)
                    .sum();
                losses.push((d / bound, cols));
            }
            let first = losses[0].0;
            losses.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let med = losses[losses.len() / 2].0;
            eprintln!(
                "w={w:3} best={:.3} median={:.3} worst={:.3} keyed={:.3} best_cols={:?}",
                losses[0].0, med, losses[losses.len() - 1].0, first, losses[0].1
            );
        }
    }

    fn load(name: &str) -> Vec<u8> {
        std::fs::read(format!("test-vectors/image/{name}")).unwrap()
    }

    /// The all-AC cost pass prices zeros too (every AC coefficient finite),
    /// DC is not, and the non-zero AC costs equal the CostMap's.
    #[test]
    fn ac_costs_cover_every_ac_coefficient() {
        use crate::stego::cost::uniward::{compute_ac_costs_streaming, compute_uniward};
        let img = JpegImage::from_bytes(&load("photo_640x480_q75_420.jpg")).unwrap();
        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let qt = img.quant_table(qt_id).unwrap();
        let all = compute_ac_costs_streaming(img.dct_grid(0), qt).unwrap();
        let map = compute_uniward(img.dct_grid(0), qt);
        let g = img.dct_grid(0);
        let bw = g.blocks_wide();
        let mut nz = 0;
        for (f, &c) in all.iter().enumerate() {
            let (blk, k) = (f / 64, f % 64);
            assert_eq!(c.is_finite(), k != 0, "flat {f}");
            if k != 0 && g.coeffs()[f] != 0 {
                nz += 1;
                assert_eq!(map.get(blk / bw, blk % bw, k / 8, k % 8).to_bits(), c.to_bits());
            }
        }
        assert!(nz > 0);
    }

    /// The separable cost pass equals the direct kernel (to f32 rounding).
    #[test]
    fn separable_costs_match_direct() {
        use crate::stego::cost::uniward::compute_ac_costs_separable;
        for name in ["photo_640x480_q75_420.jpg", "photo_320x240_q75_420.jpg"] {
            let img = JpegImage::from_bytes(&load(name)).unwrap();
            let qt = img.quant_table(img.frame_info().components[0].quant_table_id as usize).unwrap();
            let t0 = std::time::Instant::now();
            let a = crate::stego::cost::uniward::compute_ac_costs_streaming(img.dct_grid(0), qt).unwrap();
            let t1 = std::time::Instant::now();
            let b = compute_ac_costs_separable(img.dct_grid(0), qt).unwrap();
            let t2 = std::time::Instant::now();
            let mut worst = 0.0f64;
            for (x, y) in a.iter().zip(&b) {
                if x.is_finite() || y.is_finite() {
                    worst = worst.max(((*x as f64) - (*y as f64)).abs() / (*x as f64));
                }
            }
            eprintln!("{name}: direct {:?} separable {:?} worst rel diff {worst:e}", t1 - t0, t2 - t1);
            assert!(worst < 1e-5, "{name}: {worst}");
        }
    }

    fn roundtrip(name: &str, msg_len: usize) {
        use crate::stego::ghost::pipeline::{ghost_decode, ghost_encode};
        let cover = load(name);
        let msg: String = (0..msg_len).map(|i| (b'a' + ((i * 7 + i / 13) % 26) as u8) as char).collect();
        let stego = ghost_encode(&cover, &msg, "pass-v3").unwrap();
        let out = ghost_decode(&stego, "pass-v3").unwrap();
        assert_eq!(out.text, msg, "{name} len {msg_len}");
        assert!(ghost_decode(&stego, "wrong-v3").is_err());
    }

    #[test]
    fn roundtrip_primary() {
        roundtrip("photo_320x240_q75_420.jpg", 5);
        roundtrip("photo_320x240_q75_420.jpg", 300);
        roundtrip("photo_640x480_q75_420.jpg", 40);
        roundtrip("photo_640x480_q75_420.jpg", 2000);
    }

    /// Random (incompressible) payload far beyond v2's capacity still fits
    /// and round-trips (v3's ceiling is one bit per AC coefficient).
    #[test]
    fn roundtrip_heavy_payload() {
        use crate::stego::ghost::pipeline::{ghost_decode, ghost_encode};
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        let cover = load("photo_320x240_q75_420.jpg");
        let img = JpegImage::from_bytes(&cover).unwrap();
        let v2_cap = crate::stego::ghost::capacity::estimate_capacity(&img).unwrap();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(5);
        let msg: String = (0..v2_cap * 2).map(|_| (b'!' + rng.gen08::<u8>() % 90) as char).collect();
        let stego = ghost_encode(&cover, &msg, "heavy").unwrap();
        assert_eq!(ghost_decode(&stego, "heavy").unwrap().text, msg);
    }

    /// v3 changes only the Y AC coefficients, by ±1, never DC, never chroma.
    #[test]
    fn changes_are_plus_minus_one_on_y_ac() {
        use crate::stego::ghost::pipeline::ghost_encode;
        let cover = load("photo_640x480_q75_420.jpg");
        let stego = ghost_encode(&cover, &"x".repeat(500), "pm1").unwrap();
        let a = JpegImage::from_bytes(&cover).unwrap();
        let b = JpegImage::from_bytes(&stego).unwrap();
        let (ga, gb) = (a.dct_grid(0), b.dct_grid(0));
        let mut changed = 0;
        for (i, (&x, &y)) in ga.coeffs().iter().zip(gb.coeffs()).enumerate() {
            if x != y {
                assert_ne!(i % 64, 0, "DC changed");
                assert_eq!((x - y).abs(), 1);
                changed += 1;
            }
        }
        assert!(changed > 0);
        for c in 1..a.num_components() {
            assert_eq!(a.dct_grid(c).coeffs(), b.dct_grid(c).coeffs());
        }
    }

    /// The cover's own decompressed pixels, nudged, as "source" pixels for
    /// side information (RGB, 3 bytes per pixel).
    fn nudged_rgb(img: &JpegImage, f: f64) -> (Vec<u8>, u32, u32) {
        let (px, w, h) = crate::codec::jpeg::pixels::jpeg_to_luma_f64(img).unwrap();
        let rgb = px
            .iter()
            .enumerate()
            .flat_map(|(i, &y)| {
                let v = (y + ((i as f64 * f).sin() * 0.45)).round().clamp(0.0, 255.0) as u8;
                [v, v, v]
            })
            .collect();
        (rgb, w as u32, h as u32)
    }

    fn shadow_roundtrip(name: &str, n_shadows: usize, shadow_len: usize, primary_len: usize, si: bool) {
        use crate::stego::ghost::pipeline::{ghost_decode, ghost_encode_with_shadows, ghost_shadow_decode, ShadowLayer};
        let cover = load(name);
        let img = JpegImage::from_bytes(&cover).unwrap();
        let side = if si {
            let (rgb, w, h) = nudged_rgb(&img, 0.37);
            let qt = img.quant_table(img.frame_info().components[0].quant_table_id as usize).unwrap();
            Some(SideInfo::compute(&rgb, w, h, img.dct_grid(0), &qt.values).unwrap())
        } else {
            None
        };
        let primary: String = (0..primary_len).map(|i| (b'A' + (i % 26) as u8) as char).collect();
        let layers: Vec<ShadowLayer> = (0..n_shadows)
            .map(|k| ShadowLayer {
                message: (0..shadow_len).map(|i| (b'a' + ((i + k) % 26) as u8) as char).collect(),
                passphrase: format!("shadow-{k}"),
                files: vec![],
            })
            .collect();
        let stego = ghost_encode_with_shadows(&cover, &primary, &[], "primary-pass", &layers, side).unwrap();
        assert_eq!(ghost_decode(&stego, "primary-pass").unwrap().text, primary);
        for l in &layers {
            assert_eq!(ghost_shadow_decode(&stego, &l.passphrase).unwrap().text, l.message);
            let (p, _) = crate::stego::smart_decode(&stego, &l.passphrase).unwrap();
            assert_eq!(p.text, l.message);
        }
        assert!(ghost_shadow_decode(&stego, "nobody").is_err());
    }

    #[test]
    fn shadows_roundtrip_v3() {
        shadow_roundtrip("photo_640x480_q75_420.jpg", 1, 24, 200, false);
        shadow_roundtrip("photo_640x480_q75_420.jpg", 3, 64, 400, false);
        shadow_roundtrip("photo_320x240_q75_420.jpg", 2, 24, 60, false);
        shadow_roundtrip("photo_640x480_q75_420.jpg", 2, 64, 300, true);
    }

    /// A synthetic textured JPEG big enough for several lanes.
    fn synth_jpeg(w: u32, h: u32) -> Vec<u8> {
        use image::codecs::jpeg::JpegEncoder;
        let mut px = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503);
                let noise = ((v >> 9) & 0x3F) as u8;
                px.push((((x / 3) ^ (y / 5)) as u8).wrapping_add(noise));
                px.push((((x + y) / 4) as u8).wrapping_add(noise / 2));
                px.push(((x ^ y) as u8) / 2 + noise / 3);
            }
        }
        let mut buf = Vec::new();
        JpegEncoder::new_with_quality(&mut buf, 85)
            .encode(&px, w, h, image::ExtendedColorType::Rgb8)
            .unwrap();
        buf
    }

    /// Multi-lane round trip (n ≥ 2 · LANE_POSITIONS), plus a tiny message
    /// on the same large cover (one lane over everything).
    #[test]
    fn roundtrip_multi_lane() {
        use crate::stego::ghost::pipeline::{ghost_decode, ghost_encode};
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        let cover = synth_jpeg(1600, 1600);
        let img = JpegImage::from_bytes(&cover).unwrap();
        let n = img.dct_grid(0).blocks_wide() * img.dct_grid(0).blocks_tall() * 63;
        assert!(n >= 2 * LANE_POSITIONS, "n={n}");
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(11);
        let big: String = (0..3000).map(|_| (b'!' + rng.gen08::<u8>() % 90) as char).collect();
        let m_bits = (frame::FRAME_OVERHEAD + 3000) * 8; // upper bound; compression may shrink it
        assert!(lanes(n, m_bits) >= 2);
        for msg in [big.as_str(), "tiny"] {
            let stego = ghost_encode(&cover, msg, "lanes").unwrap();
            assert_eq!(ghost_decode(&stego, "lanes").unwrap().text, msg);
            assert!(ghost_decode(&stego, "not-lanes").is_err());
        }
    }

    /// The decoder's per-lane-count frame-size intervals tile every frame
    /// size, and each holds exactly its lane count.
    #[test]
    fn lane_m_ranges_tile_the_frame_sizes() {
        for n in [75_600usize, 258_048, 2_500_000, 12_002_256, 70_000_000] {
            let l_max = (n / LANE_POSITIONS).clamp(1, MAX_LANES);
            let mut next = MIN_FRAME_BITS;
            for l in 1..=l_max {
                let Some((lo, hi)) = lane_m_range(n, l) else { continue };
                assert_eq!(lo, next, "n={n} l={l}");
                assert_eq!(lanes(n, lo), l);
                assert_eq!(lanes(n, hi), l);
                if hi < n {
                    assert_eq!(lanes(n, hi + 1), l + 1);
                }
                next = hi + 1;
            }
            assert_eq!(next, n + 1, "n={n}");
        }
    }

    /// A frame over the v1 decoder's 256 KB bound (extended header) still
    /// decodes: v3 frames are bounded only by the payload format (a 48 MP
    /// photo's capacity is ~370 KB; regression of 2026-10-27, found on
    /// cgPhone).
    ///
    /// The cover is 2400 × 2400 (5.7 M positions, layer width 4), not
    /// 1600 × 1600: there this frame filled 92 % of the positions, layer
    /// width 2, where the layer-1 trellis (its layer-2 flips wet, ~16 %) is
    /// infeasible for about one frame in six — and the frame bits follow the
    /// random salt and nonce, so the test failed intermittently with
    /// `MessageTooLarge` (2026-10-31; budget- and worker-independent).
    /// Fixed salt/nonce on top, so the test cannot flake.
    #[test]
    fn roundtrip_frame_over_256k() {
        use crate::stego::crypto::DeterministicSeedGuard;
        use crate::stego::ghost::pipeline::{ghost_decode, ghost_encode};
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        let _g = DeterministicSeedGuard::set("256");
        let cover = synth_jpeg(2400, 2400);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(256);
        let alphabet = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .";
        let msg: String = (0..390_000).map(|_| alphabet[rng.gen_range08(0..64)] as char).collect();
        assert!(crate::stego::payload::compressed_payload_size(&msg, &[]) > crate::stego::frame::MAX_FRAME_BYTES);
        let stego = ghost_encode(&cover, &msg, "big-frame").unwrap();
        assert_eq!(ghost_decode(&stego, "big-frame").unwrap().text, msg);
    }

    /// A Ghost encode at the cover's maximum (the capacity bar's 100 %)
    /// writes the same bytes under any memory budget and worker count: the
    /// budget picks the memory layout (in-memory vs checkpointed trellises,
    /// cost groups, lanes at once), never the result — the decoder does not
    /// know the budget. Distinct plans are asserted so the test keeps
    /// covering several rungs.
    #[test]
    fn encode_bytes_independent_of_budget_and_workers() {
        use crate::stego::crypto::DeterministicSeedGuard;
        use crate::stego::ghost::pipeline::{ghost_decode, ghost_encode};
        use crate::stego::memory::{budget_test_lock, ghost_lane_plan, set_memory_budget, ModeId};
        let cover = synth_jpeg(1600, 1600);
        let img = JpegImage::from_bytes(&cover).unwrap();
        let n = ac_count(img.dct_grid(0));
        let max = crate::stego::ghost::capacity::ghost_bands(&img, 0, 0).unwrap().max_bytes;
        let mut x = 0x2545_f491u32;
        let mut msg: String = (0..max)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"[(x % 62) as usize] as char
            })
            .collect();
        while crate::stego::payload::compressed_payload_size(&msg, &[]) > max {
            msg.pop();
        }
        let m = (crate::stego::payload::compressed_payload_size(&msg, &[]) + frame::FRAME_OVERHEAD) * 8;
        assert!(lanes(n, m) >= 2, "n={n} m={m}");

        let _b = budget_test_lock();
        let _s = DeterministicSeedGuard::set("1031");
        // (workers, budget): unlimited; 1 MB (one checkpointed lane per cost
        // group); 200 MB (in-memory lanes with a smaller table on one
        // worker, checkpointed lanes on several).
        #[cfg(feature = "parallel")]
        let workers = [1usize, 4];
        #[cfg(not(feature = "parallel"))]
        let workers = [1usize];
        let budgets = [None, Some(1usize << 20), Some(200 << 20)];
        let mut base: Option<Vec<u8>> = None;
        let mut plans = Vec::new();
        for &k in &workers {
            #[cfg(feature = "parallel")]
            let pool = rayon::ThreadPoolBuilder::new().num_threads(k).build().unwrap();
            let run = |f: &(dyn Fn() -> Vec<u8> + Sync)| -> Vec<u8> {
                #[cfg(feature = "parallel")]
                return pool.install(f);
                #[cfg(not(feature = "parallel"))]
                return f();
            };
            for &b in &budgets {
                set_memory_budget(b);
                let p = run(&|| {
                    let p = ghost_lane_plan(ModeId::GhostNoShadow, 1600, 1600, n, m, V31_STC_H);
                    format!("{:?}/{}/{}/{}", p.rung, p.parallel_lanes, p.lane_inline_bytes, p.group_positions).into_bytes()
                });
                let out = run(&|| ghost_encode(&cover, &msg, "budget").unwrap());
                set_memory_budget(None);
                plans.push(String::from_utf8(p).unwrap());
                match &base {
                    None => {
                        assert_eq!(ghost_decode(&out, "budget").unwrap().text, msg);
                        base = Some(out);
                    }
                    Some(want) => assert!(&out == want, "workers {k} budget {b:?} ({}): bytes differ", plans.last().unwrap()),
                }
            }
        }
        plans.sort();
        plans.dedup();
        assert!(plans.len() >= 3, "too few distinct plans: {plans:?}");
    }

    /// Peak lane memory does not grow as the message shrinks: at every frame
    /// size the lanes of a batch (as `plan` runs them, on 4 or 6 workers)
    /// stay within the large-frame case, because longer lanes keep only their
    /// share of a table and recompute the rest. (The one exception is the
    /// lane vectors alone, 8 B per position: one lane over a whole 48 MP
    /// cover holds 384 MB of them, over the 4-worker large-frame case.)
    #[test]
    fn lane_memory_does_not_grow_for_small_frames() {
        for workers in [4usize, 6] {
            for n in [12_002_256usize, 48_009_024] {
                // (lanes' working set, their vectors alone)
                let total = |m: usize| {
                    let l = lanes(n, m);
                    let k = workers.min(l);
                    let nk = share(n, l, 0).len();
                    let inline = crate::stego::memory::LANE_INLINE_BYTES.min(lane_inline_share(n, nk, k, V31_STC_H, workers));
                    (k * lane_bytes(nk, m.div_ceil(l), V31_STC_H, inline, lane_threads(k, workers)), k * nk * 8)
                };
                let full = [n / 16, n / 8, n / 5].into_iter().map(|m| total(m).0).max().unwrap();
                // Checkpoints and the wave buffers of the recomputed part
                // (one message block each: 12 MB for a 64-byte frame on 48 MP).
                let slack = n * 2;
                for m in [500usize, 4_000, 9_000, 25_000, 50_000, 400_000] {
                    let (t, vectors) = total(m);
                    assert!(t <= full.max(vectors) + slack, "workers {workers} n={n} m={m}: lanes {t} B > full-lane {full} B");
                }
            }
        }
    }

    /// The plan does not depend on the memory knobs: cost groups (one lane
    /// per group vs everything at once), lanes at once, in-memory vs
    /// checkpointed trellis.
    #[test]
    fn plan_is_independent_of_memory_knobs() {
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        let cover = JpegImage::from_bytes(&synth_jpeg(1600, 1600)).unwrap();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9);
        let key = [0x33u8; 64];
        for m in [3_000usize, 40_000] {
            let bits: Vec<u8> = (0..m).map(|_| rng.gen08::<u8>() & 1).collect();
            let base = plan(&cover, &bits, &key, None, None, None, &EncodeOpts::default()).unwrap();
            let cases = [
                EncodeOpts { group_positions: 1, ..Default::default() },
                EncodeOpts { parallel_lanes: 1, lane_inline_bytes: 0, ..Default::default() },
                EncodeOpts { group_positions: 1_500_000, parallel_lanes: 2, ..Default::default() },
                EncodeOpts { lane_inline_bytes: 20 << 20, ..Default::default() },
                EncodeOpts { lane_inline_bytes: 1 << 20, parallel_lanes: 1, ..Default::default() },
            ];
            for opts in &cases {
                let p = plan(&cover, &bits, &key, None, None, None, opts).unwrap();
                assert_eq!(p.changes, base.changes, "m={m} {opts:?}");
                assert_eq!(p.stats.distortion.to_bits(), base.stats.distortion.to_bits());
            }
            // One worker: no waves, the trellis keeps a tail of its share
            // instead (and an in-memory table in the default case).
            #[cfg(feature = "parallel")]
            {
                let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
                for opts in [&EncodeOpts::default(), &cases[3], &cases[4]] {
                    let p = one.install(|| plan(&cover, &bits, &key, None, None, None, opts)).unwrap();
                    assert_eq!(p.changes, base.changes, "m={m} one worker {opts:?}");
                    assert_eq!(p.stats.distortion.to_bits(), base.stats.distortion.to_bits());
                }
            }
        }
    }

    #[test]
    fn lane_geometry() {
        assert_eq!(lanes(258_048, 10_000), 1);
        assert_eq!(lanes(12_002_256, 400), 1);
        assert_eq!(lanes(12_002_256, 600_000), 11);
        assert_eq!(lanes(12_002_256, 20_000), 4);
        assert_eq!(lanes(1 << 40, 1 << 40), MAX_LANES);
        // shares tile the range exactly
        for (total, l) in [(12_002_256usize, 11usize), (1000, 7), (5, 3)] {
            let mut next = 0;
            for k in 0..l {
                let r = share(total, l, k);
                assert_eq!(r.start, next);
                next = r.end;
            }
            assert_eq!(next, total);
        }
        assert_eq!(layer_split(9), (5, 4));
        assert_eq!(layer_split(408), (204, 204));
    }

    /// v3 encode is deterministic (fixed salt/nonce) and independent of the
    /// lane scheduling; the hash is pinned so a platform or refactor that
    /// changes the output is noticed (aarch64 value; x86_64 / WASM must match).
    #[test]
    fn encode_is_deterministic_and_pinned() {
        use crate::stego::crypto::DeterministicSeedGuard;
        use crate::stego::ghost::pipeline::ghost_encode;
        use sha2::{Digest, Sha256};
        let cover = load("photo_640x480_q75_420.jpg");
        let _g = DeterministicSeedGuard::set("2026");
        let a = ghost_encode(&cover, &"pinned message ".repeat(40), "pin").unwrap();
        let b = ghost_encode(&cover, &"pinned message ".repeat(40), "pin").unwrap();
        assert_eq!(a, b);
        let h: String = Sha256::digest(&a).iter().map(|x| format!("{x:02x}")).collect();
        eprintln!("v3 pin {h}");
        if let Ok(want) = std::env::var("PHASM_V3_PIN") {
            assert_eq!(h, want);
        }
    }

    /// The split encode (`session.rs`: cost bands, lane layers, split
    /// trellises with segments traced on their helpers, results out of
    /// order) gives exactly `ghost_encode_with_files`'s bytes, for any
    /// number of helpers, with and without Deep Cover side information.
    #[test]
    fn split_encode_is_byte_identical() {
        use crate::stego::crypto::DeterministicSeedGuard;
        use crate::stego::ghost::pipeline::{ghost_encode_si_with_files, ghost_encode_with_files};
        use crate::stego::ghost::session::run_split_encode_locally;
        use crate::stego::payload::FileEntry;
        let _g = DeterministicSeedGuard::set("31");
        let small = load("photo_640x480_q75_420.jpg");
        let big = synth_jpeg(2400, 1800);
        let files = vec![FileEntry { filename: "a.bin".into(), content: (0..900u32).map(|i| (i * 7) as u8).collect() }];
        // Incompressible text of `n` characters.
        let noise = |n: usize| -> String {
            let mut x = 0x2545f491u32;
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"[(x % 62) as usize] as char
                })
                .collect()
        };
        // (cover, message, files, helper counts, split trellises expected
        // at the largest count)
        for (cover, msg, fl, helper_counts, split) in [
            (&small, "short".to_string(), &[][..], &[1usize, 2, 5][..], false), // one in-memory lane
            (&small, noise(600), &files[..], &[1, 3][..], false),
            (&big, noise(3200), &[][..], &[3][..], false), // 4 full-size lanes, in-memory trellises
            (&big, noise(1500), &[][..], &[2, 3][..], true), // 2 long lanes: whole (2 helpers), split (3)
            (&big, "tiny".to_string(), &[][..], &[3][..], true), // one lane over the whole cover
        ] {
            let want = ghost_encode_with_files(cover, &msg, fl, "split").unwrap();
            for &helpers in helper_counts {
                let (got, _, counts) = run_split_encode_locally(cover, &msg, fl, "split", None, helpers, None).unwrap();
                assert!(got == want, "helpers {helpers}, msg len {}", msg.len());
                if helpers == *helper_counts.last().unwrap() {
                    assert_eq!(counts[3] > 0 && counts[5] > 0, split, "split use, msg len {} {counts:?}", msg.len());
                }
            }
            if split {
                // Pass A in the coordinator (the large-cover rule, forced).
                let (got, _, counts) = run_split_encode_locally(cover, &msg, fl, "split", None, 3, Some(0)).unwrap();
                assert!(got == want, "local pass A, msg len {}", msg.len());
                assert!(counts[0] > 0 && counts[3] == 0 && counts[5] > 0, "local pass A use {counts:?}");
            }
        }
        let img = JpegImage::from_bytes(&small).unwrap();
        let (rgb, w, h) = nudged_rgb(&img, 0.4);
        let want = ghost_encode_si_with_files(&small, &rgb, w, h, &"si ".repeat(50), &[], "split").unwrap();
        for helpers in [1usize, 4] {
            let (got, _, _) = run_split_encode_locally(&small, &"si ".repeat(50), &[], "split", Some((&rgb, w, h)), helpers, None).unwrap();
            assert!(got == want, "si helpers {helpers}");
        }
    }

    /// Deep Cover (SI) primary round trip.
    #[test]
    fn roundtrip_si_primary() {
        use crate::stego::ghost::pipeline::{ghost_decode, ghost_encode_si};
        let cover = load("photo_640x480_q75_420.jpg");
        let img = JpegImage::from_bytes(&cover).unwrap();
        let (rgb, w, h) = nudged_rgb(&img, 0.61);
        let msg = "deep cover v3 ".repeat(60);
        let stego = ghost_encode_si(&cover, &rgb, w, h, &msg, "si-pass").unwrap();
        assert_eq!(ghost_decode(&stego, "si-pass").unwrap().text, msg);
    }

    /// Diagnostic: coding loss vs constraint height h (scalar reference
    /// Viterbi, any h ≤ 12), end-set random H-hat, on a heavy-tailed
    /// (log-uniform over 8 octaves) cost profile, binary.
    /// `cargo test --release stc_height_sweep -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn stc_height_sweep() {
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        fn viterbi(cover: &[u8], cost: &[f64], msg: &[u8], cols: &[u32], h: usize) -> f64 {
            let w = cols.len();
            let ns = 1usize << h;
            let mut prev = vec![f64::INFINITY; ns];
            prev[0] = 0.0;
            let mut cur = vec![0.0f64; ns];
            let mut mi = 0;
            for j in 0..cover.len() {
                let col = cols[j % w] as usize;
                let (c0, c1) = if cover[j] == 0 { (0.0, cost[j]) } else { (cost[j], 0.0) };
                for s in 0..ns {
                    let a = prev[s] + c0;
                    let b = prev[s ^ col] + c1;
                    cur[s] = if b < a { b } else { a };
                }
                std::mem::swap(&mut prev, &mut cur);
                if j % w == w - 1 && mi < msg.len() {
                    let bit = msg[mi] as usize;
                    for s in 0..ns / 2 {
                        cur[s] = prev[(s << 1) | bit];
                    }
                    for s in ns / 2..ns {
                        cur[s] = f64::INFINITY;
                    }
                    std::mem::swap(&mut prev, &mut cur);
                    mi += 1;
                }
            }
            prev.iter().cloned().fold(f64::INFINITY, f64::min)
        }
        let n = 40_000usize;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(3);
        let costs: Vec<f64> = (0..n).map(|_| (rng.gen08::<f64>() * 8.0 * core::f64::consts::LN_2).exp()).collect();
        let cover: Vec<u8> = (0..n).map(|_| rng.gen08::<u8>() & 1).collect();
        let msg: Vec<u8> = (0..n).map(|_| rng.gen08::<u8>() & 1).collect();
        for inv_alpha in [2usize, 5, 20, 50] {
            let m = n / inv_alpha;
            let ab: Vec<(f32, f32)> = costs.iter().map(|&c| (c as f32, f32::INFINITY)).collect();
            let b = bound_of(&ab, n, m);
            let mut line = format!("1/alpha={inv_alpha:3}");
            for h in [7usize, 8, 9, 10, 11] {
                let mut best = f64::INFINITY;
                let mut sum = 0.0;
                for t in 0..3 {
                    let cols: Vec<u32> = (0..inv_alpha).map(|_| (rng.gen08::<u32>() & ((1 << h) - 1)) | 1 | (1 << (h - 1))).collect();
                    let d = viterbi(&cover, &costs, &msg[..m], &cols, h);
                    best = best.min(d);
                    sum += d;
                    let _ = t;
                }
                line += &format!("  h={h}: {:.3}", sum / 3.0 / b);
            }
            eprintln!("{line}");
        }
    }

    /// PHASE-1 measurement: distortion and changes of the production v3
    /// encoder at STC heights 7/8/9 on real covers. Random frame bits of
    /// `rate · nzAC` (frame bpnzAC) per cover and rate, identical for every h.
    /// `V3_SWEEP_COVERS=<dir of JPEGs> V3_SWEEP_N=50 cargo test --release
    /// height_sweep_covers -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn height_sweep_covers() {
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        let dir = std::env::var("V3_SWEEP_COVERS").expect("V3_SWEEP_COVERS");
        let n_covers: usize = std::env::var("V3_SWEEP_N").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
        let heights: Vec<usize> = std::env::var("V3_SWEEP_H")
            .ok()
            .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
            .unwrap_or(vec![7, 8, 9]);
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .filter(|n| n.ends_with(".jpg"))
            .collect();
        names.sort_by_key(|n| n.trim_end_matches(".jpg").parse::<u64>().unwrap_or(u64::MAX));
        names.truncate(n_covers);
        let key = [0x5au8; 64];
        let rates = [0.05f64, 0.10, 0.20];
        println!("cover\trate\th\tnzac\tm\tchanges\tdistortion\tsecs");
        for (ci, name) in names.iter().enumerate() {
            let img = JpegImage::from_bytes(&std::fs::read(format!("{dir}/{name}")).unwrap()).unwrap();
            let g = img.dct_grid(0);
            let nz = g.coeffs().iter().enumerate().filter(|(i, c)| i % 64 != 0 && **c != 0).count();
            for (ri, &rate) in rates.iter().enumerate() {
                let m = ((rate * nz as f64) as usize).max(64);
                let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64((ci * 16 + ri) as u64);
                let bits: Vec<u8> = (0..m).map(|_| rng.gen08::<u8>() & 1).collect();
                for &h in &heights {
                    let mut im = img.clone();
                    let t0 = std::time::Instant::now();
                    let opts = EncodeOpts { h, ..Default::default() };
                    let st = encode(&mut im, &bits, &key, None, &opts).unwrap();
                    let secs = t0.elapsed().as_secs_f64();
                    println!("{name}\t{rate}\t{h}\t{nz}\t{m}\t{}\t{:.3}\t{secs:.3}", st.changes, st.distortion);
                }
            }
        }
    }

    #[test]
    fn shuffle_matches_coeffpos_permutation() {
        use crate::stego::permute::{permute_positions, CoeffPos};
        let seed = [7u8; 32];
        let mut a: Vec<u32> = (0..5000).collect();
        let mut b: Vec<CoeffPos> = (0..5000).map(|i| CoeffPos { flat_idx: i, cost: 0.0 }).collect();
        shuffle_u32(&mut a, &seed);
        permute_positions(&mut b, &seed);
        assert!(a.iter().zip(&b).all(|(x, y)| *x == y.flat_idx));
    }

    /// One shuffle reads both generations: v3.1's search finds only v3.1
    /// frames, `decode_all` finds either.
    #[test]
    fn decode_all_reads_v31_then_v3() {
        let pass = "wire-format-cross-platform-test";
        let key = crate::stego::crypto::derive_structural_key(pass).unwrap();
        let parse = &crate::stego::ghost::pipeline::try_parse_and_decrypt;
        for (name, is_v31) in [("ghost_v31_2026_11_03.jpg", true), ("ghost_v3_2026_11_03.jpg", false)] {
            let img = JpegImage::from_bytes(&load(&format!("wire-format/{name}"))).unwrap();
            let planes = decode_planes(&img, &key);
            assert_eq!(decode_v31(&planes, pass, &key, parse).is_some(), is_v31, "{name}");
            let got = decode_all(&img, pass, &key, parse).expect("a frame").unwrap();
            assert_eq!(got.text, "Hello from the wire-format gate!", "{name}");
        }
    }

    #[test]
    fn si_cost_policy() {
        let t = |e: f32, exact: bool, tie: bool| Some(SiTerm { e, exact, tie });
        // Exact: ρ(1 − 2|e|) toward the precover, no floor, ρ away.
        let (rp, rm) = pm_costs(10.0, 3, t(0.45, true, false));
        assert!((rp - 1.0).abs() < 1e-5 && rm == 10.0);
        let (rp, rm) = pm_costs(10.0, 0, t(-0.45, true, false));
        assert!(rp == 10.0 && (rm - 1.0).abs() < 1e-5);
        // Exact tie: unmodulated.
        assert_eq!(pm_costs(10.0, 2, t(0.5, true, true)), (10.0, 10.0));
        // Approximate: factor floored at 0.3.
        let (rp, _) = pm_costs(10.0, 3, t(0.45, false, false));
        assert!((rp - 3.0).abs() < 1e-5);
        // No side information / zero error: symmetric.
        assert_eq!(pm_costs(10.0, 3, None), (10.0, 10.0));
        assert_eq!(pm_costs(10.0, 3, t(0.0, true, false)), (10.0, 10.0));
    }
}
