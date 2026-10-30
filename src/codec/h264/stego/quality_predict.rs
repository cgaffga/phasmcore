// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Per-clip capacity maximum (spec `stego-cover-order-and-damage-costs.md`
//! § 4a, "adaptive red end"): predict the visual damage of an embed at a
//! change rate α from data the capacity probe already has, and raise the
//! red end above the fixed floor `CAP_ALPHA_MAX` (1/8) for clips that
//! tolerate it.
//!
//! **Predictor.** Per GOP, the syntax-dry STC cover is binned by the damage
//! term `D` (the ranking the encoder's costs follow, `cost = W · S · D`;
//! the content term `S` is left out — it needs the YUV and a wavelet pass the
//! probe skips, and it did not predict better). For a payload of `α · n` bits
//! the flips are modelled by the payload-limited sender (Gibbs):
//! `p = 1 / (1 + e^{λ ρ})` with `Σ h(p) = α n` over the STC's used cover
//! (`⌊n / m⌋ · m` positions). The predicted damage is
//! `E(α) = Σ p · e · frames_left` — the flip's own luma energy, persisted to
//! the end of its key-frame segment ([`DamageModel::predictor_terms`]).
//!
//! **Calibration** (`docs/design/video/h264/intra-drift-diagnosis.md`,
//! "Follow-up 4"): on the 12 corpus cells × 9 α × 3 passphrases, with the
//! apps' planner, the measured clip PSNR-Y and worst-MB PSNR-Y fit
//! `P = a + b · 10 log10(E / pixels) + c · 10 log10(α)` (the α term carries
//! the drift compounding the linear energy sum misses). The fit is good at
//! telling a dense, hard clip from the rest but its per-clip level error is
//! ~2.5–3 dB and every encode adds its own ±2–5 dB lottery (random salt →
//! different flips). So the predictor is used for the one call it makes
//! reliably — *dense, hard clip* vs *the rest* — as a **two-level** red end:
//! [`ALPHA64_CAP`] / 64 = 3/16 when the planned embed at 3/16 is predicted to
//! clear both floors with the margins [`CLIP_MARGIN_DB`] / [`WMB_MARGIN_DB`],
//! else the fixed 1/8. Finer steps are not resolvable: the hardest clip
//! (iphone5 1080p) is predicted to clear the floor at 10/64 and measured
//! below it on 1 of 8 encodes. Above 3/16 the corpus showed lottery failures
//! and flood frames on clips the predictor (and any single trial embed)
//! rated easy.
//!
//! **Determinism.** The binning, the energies and the sums are integers; the
//! Gibbs solve and the logarithms use only IEEE `+ − × ÷`,
//! [`crate::det_math::det_exp`] and the series [`det_ln`], with fixed
//! iteration counts — the figure is the same on iOS, Android and the CLI for
//! the same cover. It is a displayed value only; nothing is persisted.

use super::cost_weights::coeff_is_dry;
use super::damage::DamageModel;
use super::inject::DomainCover;
use crate::det_math::det_exp;

/// Grid of the per-GOP curve: `α_k = k / 64`, `k = 1..=Q_GRID` (up to 5/16;
/// the apps' planner can put a GOP above the clip α — about 0.24 for long
/// clips — so the curve reaches past the cap).
pub const Q_GRID: usize = 20;
/// The red end never drops below the fixed `CAP_ALPHA_MAX` = 1/8 = 8/64.
pub const ALPHA64_FLOOR: u32 = 8;
/// …and the raised level is 3/16 = 12/64 (two levels, see the module docs).
pub const ALPHA64_CAP: u32 = 12;

/// The maintainer's picture-quality floor (unchanged): clip PSNR-Y and
/// worst-MB PSNR-Y against the natural encode.
pub const CLIP_FLOOR_DB: f64 = 27.0;
pub const WMB_FLOOR_DB: f64 = 13.0;
/// Safety margins on the predicted figures at 3/16. In-sample and
/// leave-one-clip-out they put every iphone5 cell at 1/8 and every other
/// corpus cell at 3/16, with ≥ 1.1 dB slack in-sample on each decision.
pub const CLIP_MARGIN_DB: f64 = 5.5;
pub const WMB_MARGIN_DB: f64 = 4.5;

/// `P_clip = CLIP_FIT[0] + CLIP_FIT[1] · x + CLIP_FIT[2] · la`, `x` = 10 log10
/// of the predicted clip MSE, `la` = 10 log10 α.
pub const CLIP_FIT: [f64; 3] = [31.598, -0.711, -0.939];
/// The same for the worst-MB PSNR (`x` = the worst stego GOP's MSE; fitted on
/// 2-GOP clips with `x` pooled over the stego window, which agrees within
/// 0.5 dB there — see [`predict_quality`]).
pub const WMB_FIT: [f64; 3] = [13.271, -0.330, -1.004];

const LN2: f64 = 0.6931471805599453;
const INV_LN10: f64 = 0.43429448190325176;
/// `⌊2^(j/8) · 2^24⌋`, `j = 1..=7` — fractional eighth-octave bin edges.
const OCT8_EDGES: [u64; 7] = [18295683, 19951584, 21757357, 23726566, 25874004, 28215801, 30769549];
const N_BINS: usize = 128 * 8;

/// Predicted damage curve of one GOP.
#[derive(Clone, Debug, PartialEq)]
pub struct GopQualityCurve {
    /// `E(α_k)` in px² (expected persisted luma squared error, summed over
    /// the GOP's frames), `k = 1..=Q_GRID`.
    pub energy: [f64; Q_GRID],
    /// Syntax-dry STC positions of the GOP.
    pub n_dry: usize,
    pub n_frames: u32,
    /// Predicted share of the flips at `α_k` that land in intra-coded MBs
    /// of non-key frames (intra MBs of P frames), `k = 1..=Q_GRID` — the
    /// damage-aware planner's hot-GOP signal ([`gop_damage_weight`]).
    pub intra_p_share: [f64; Q_GRID],
    /// Predicted flips at `α_k` in intra MBs of non-key frames, summed over
    /// the GOP, per MB of the frame grid — the intra flood a GOP accumulates
    /// before its next key frame, which the damage-aware planner bounds
    /// ([`gop_damage_weight`]).
    pub intra_p_per_mb: [f64; Q_GRID],
}

impl GopQualityCurve {
    /// `E(α)`, linear between grid points (0 at α = 0, clamped at the top).
    pub fn energy_at(&self, alpha: f64) -> f64 {
        if alpha <= 0.0 {
            return 0.0;
        }
        let t = alpha * 64.0;
        let k = t as usize; // floor (t ≥ 0)
        if k >= Q_GRID {
            return self.energy[Q_GRID - 1];
        }
        let lo = if k == 0 { 0.0 } else { self.energy[k - 1] };
        let hi = self.energy[k];
        lo + (hi - lo) * (t - k as f64)
    }
}

/// Eighth-octave bin of a positive integer: `⌊8 · log2 d⌋`, exactly.
fn oct8_bin(d: u128) -> usize {
    let d = d.max(1);
    let l = 127 - d.leading_zeros() as usize;
    let mant = if l >= 24 { (d >> (l - 24)) as u64 } else { (d << (24 - l)) as u64 };
    let j = OCT8_EDGES.iter().filter(|&&e| mant >= e).count();
    l * 8 + j
}

/// `ln(1 + y)` for `y ∈ [0, 1]`: `2 · atanh(y / (2 + y))`, `|z| ≤ 1/3`.
fn ln1p_unit(y: f64) -> f64 {
    let z = y / (2.0 + y);
    let z2 = z * z;
    let mut term = z;
    let mut sum = 0.0;
    let mut k = 1.0;
    for _ in 0..30 {
        sum += term / k;
        term *= z2;
        k += 2.0;
    }
    2.0 * sum
}

/// Deterministic natural log for `x > 0` (exponent split + atanh series).
pub fn det_ln(x: f64) -> f64 {
    if !(x > 0.0) {
        return f64::NEG_INFINITY;
    }
    let mut x = x;
    let mut e: i64 = 0;
    if x < f64::MIN_POSITIVE {
        x *= 4503599627370496.0; // 2^52
        e -= 52;
    }
    let bits = x.to_bits();
    e += ((bits >> 52) & 0x7ff) as i64 - 1023;
    // Mantissa in [1, 2); fold to [√½, √2) so the series argument stays small.
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > 1.4142135623730951 {
        m *= 0.5;
        e += 1;
    }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let mut term = z;
    let mut sum = 0.0;
    let mut k = 1.0;
    for _ in 0..30 {
        sum += term / k;
        term *= z2;
        k += 2.0;
    }
    e as f64 * LN2 + 2.0 * sum
}

fn det_db(x: f64) -> f64 {
    10.0 * det_ln(x) * INV_LN10
}

/// `2^q` for an integer `q` in the normal range, exactly.
fn pow2i(q: i64) -> f64 {
    f64::from_bits(((q + 1023).clamp(1, 2046) as u64) << 52)
}

/// The predicted damage curve of one walked GOP cover.
pub fn gop_quality_curve(cover: &DomainCover, n_frames: u32) -> GopQualityCurve {
    let _prof = crate::codec::h264::profile::scope("v2.quality_curve");
    let model = DamageModel::from_cover(cover);
    let mut count = vec![0u64; N_BINS];
    let mut delta = vec![0u128; N_BINS];
    let mut count_ip = vec![0u64; N_BINS];
    let mut n = 0usize;
    for d in [&cover.coeff_sign_bypass, &cover.coeff_suffix_lsb] {
        for (&key, &mag) in d.positions.iter().zip(&d.magnitudes) {
            if !coeff_is_dry(key) {
                continue;
            }
            n += 1;
            // Every dry position is priced; an unpriced one (only possible
            // on a synthetic cover without MB syntax) ranks as the cheapest.
            let (dmg, e) = model.predictor_terms(key, mag).unwrap_or((1, 0));
            let b = oct8_bin(dmg);
            count[b] += 1;
            delta[b] = delta[b].saturating_add(e);
            if model.intra_non_key(key) {
                count_ip[b] += 1;
            }
        }
    }
    let mut energy = [0.0f64; Q_GRID];
    let mut intra_p_share = [0.0f64; Q_GRID];
    let mut intra_p_per_mb = [0.0f64; Q_GRID];
    let used: Vec<usize> = (0..N_BINS).filter(|&b| count[b] > 0).collect();
    if n == 0 || used.is_empty() {
        return GopQualityCurve { energy, n_dry: n, n_frames, intra_p_share, intra_p_per_mb };
    }
    let n_mb = {
        let (w, h) = model.mb_dims();
        (w as f64 * h as f64).max(1.0)
    };
    let b0 = used[0] as i64;
    // ρ relative to the cheapest bin (scale-free: λ absorbs it), bin centre.
    const OCT8: [f64; 8] = [
        1.0,
        1.0905077326652577,
        1.189207115002721,
        1.2968395546510096,
        1.4142135623730951,
        1.5422108254079407,
        1.681792830507429,
        1.8340080864093424,
    ];
    let rho: Vec<f64> = used
        .iter()
        .map(|&b| {
            let r = b as i64 - b0;
            pow2i(r.div_euclid(8)) * OCT8[r.rem_euclid(8) as usize] * 1.0442737824274138
        })
        .collect();
    let cnt: Vec<f64> = used.iter().map(|&b| count[b] as f64).collect();
    let dlt: Vec<f64> = used.iter().map(|&b| delta[b] as f64 / 65536.0).collect();
    let cip: Vec<f64> = used.iter().map(|&b| count_ip[b] as f64).collect();
    for (ki, slot) in energy.iter_mut().enumerate() {
        let k = ki + 1;
        let m = n * k / 64;
        if m == 0 {
            continue;
        }
        // The STC runs over the first ⌊n / m⌋ · m keyed positions — a random
        // fraction `f` of the cover at local rate m / (f n).
        let w = (n / m).max(1);
        let f = (m * w) as f64 / n as f64;
        let m_eff = m as f64 / f;
        let flip_p = |lam: f64, i: usize| -> (f64, f64) {
            let x = lam * rho[i];
            if x > 700.0 {
                return (0.0, 0.0);
            }
            let y = det_exp(-x);
            let p = y / (1.0 + y);
            (p, (ln1p_unit(y) + x * p) / LN2)
        };
        // Bisection on log2 λ: Σ h(p) falls as λ grows.
        let (mut lo, mut hi) = (-240.0f64, 64.0f64);
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            let lam = det_exp(mid * LN2);
            let h: f64 = (0..used.len()).map(|i| cnt[i] * flip_p(lam, i).1).sum();
            if h > m_eff {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let lam = det_exp(hi * LN2);
        let (mut e, mut flips, mut flips_ip) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..used.len() {
            let p = flip_p(lam, i).0;
            e += p * dlt[i];
            flips += p * cnt[i];
            flips_ip += p * cip[i];
        }
        *slot = e * f;
        if flips > 0.0 {
            intra_p_share[ki] = flips_ip / flips;
        }
        intra_p_per_mb[ki] = flips_ip * f / n_mb;
    }
    GopQualityCurve { energy, n_dry: n, n_frames, intra_p_share, intra_p_per_mb }
}

/// Damage-aware planning: the reference fill at which a GOP's intra flood is
/// judged, in 1/64ths of α — the fixed red-end floor `CAP_ALPHA_MAX` = 1/8.
pub const DMG_REF_ALPHA64: usize = 8;
/// …and the most predicted flips per MB, summed over the GOP, that may land
/// in intra MBs of P frames at that fill ([`GopQualityCurve::intra_p_per_mb`]).
pub const DMG_INTRA_P_PER_MB_MAX: f64 = 1.5;
/// Floor on the damage weight: a GOP keeps at least this share of its
/// capacity in the plan.
pub const DMG_WEIGHT_MIN: f64 = 0.25;

/// The damage-aware planner's weight of one GOP, in `[DMG_WEIGHT_MIN, 1]`.
///
/// 1 unless the GOP is intra-flooded at the reference fill
/// [`DMG_REF_ALPHA64`]: it would take more than [`DMG_INTRA_P_PER_MB_MAX`]
/// predicted flips per MB (summed over the GOP) in intra MBs of P frames.
/// Then the weight is `α* / α_ref`, where `α*` is the largest α at which the
/// GOP stays at that bound (linear on the curve's grid).
///
/// See [`quality_safe_cap`] for how it enters the plan and why.
pub fn gop_damage_weight(curve: &GopQualityCurve) -> f64 {
    let k = DMG_REF_ALPHA64.clamp(1, Q_GRID);
    let pk = &curve.intra_p_per_mb;
    if !(pk[k - 1] > DMG_INTRA_P_PER_MB_MAX) {
        return 1.0;
    }
    // `pk` is non-decreasing in α (the Gibbs flips only grow with α); find
    // the grid step [α_{j}, α_{j+1}] that crosses the bound, α_0 = 0, pk = 0.
    let mut a_star = 0.0f64;
    for j in 0..k {
        let lo = if j == 0 { 0.0 } else { pk[j - 1] };
        let hi = pk[j];
        if hi > DMG_INTRA_P_PER_MB_MAX {
            let t = if hi > lo { (DMG_INTRA_P_PER_MB_MAX - lo) / (hi - lo) } else { 0.0 };
            a_star = (j as f64 + t.clamp(0.0, 1.0)) / 64.0;
            break;
        }
    }
    (a_star * 64.0 / k as f64).clamp(DMG_WEIGHT_MIN, 1.0)
}

/// The quality-safe per-GOP cap the probe hands the planner:
/// `⌊cap × gop_damage_weight⌋` (never above `cap`, the deliverable bytes).
/// The probe scales the GOP's syntax-dry count the same way for the α
/// bands (`CapacityProbeResult::dry_cover_bits`), so at the red end the
/// other GOPs sit at `α_max` and the hot one at `w · α_max`; the displayed
/// maximum shrinks by the weighted share instead of the other GOPs running
/// hotter.
///
/// **Why it is a cap and not a weight array.** `plan_safe_balanced` splits
/// the message over the stego window proportional to the caps it is given,
/// and never above them. Scaling a hot GOP's cap by `w` therefore moves
/// payload off it in the same proportion at every fill (the other GOPs
/// carry the difference, each at the same relative rate), and every place
/// the caps flow through — the planner FFI on iOS/Android, the CLI, the
/// shadow streaming encode, the capacity figure (`max_message_len`) and the
/// probe's early exit — keeps working unchanged, because a smaller cap is
/// always a valid ceiling for the encode. Per GOP and from its own walk
/// only, so a prefix probe, a probe resumed after an early Encode tap and
/// the full probe give the same value for the same GOP.
///
/// **Why intra MBs of P frames.** On the 20-GOP IMG_4138 720p clip at the
/// fixed 1/8, every one of 7 encodes put its worst MB in GOP 15 or 16
/// (10.9–14.4 dB against ≥ 15.7 dB elsewhere). Those are the only GOPs
/// where the cost ordering cannot steer around intra MBs of P frames: 40 %
/// and 72 % of their flips land there (≤ 14 % in every other GOP of the
/// clip; a fast-motion burst the encoder codes mostly intra). A flip there
/// floods through intra prediction and the next P frames reference it, so
/// the flood accumulates until the key frame; the energy predictor and the
/// per-GOP caps do not see it, and an α-even split loads those GOPs like
/// any other. The predicted figure separates them cleanly: 4.1 and 5.5
/// intra-P flips per MB at 1/8, against ≤ 0.57 for every other GOP of the
/// clip and ≤ 0.16 for every GOP of the rest of the corpus — so the bound
/// [`DMG_INTRA_P_PER_MB_MAX`] = 1.5 leaves every other clip's plan unchanged
/// (`intra-drift-diagnosis.md`, follow-up 6).
pub fn quality_safe_cap(cap: usize, curve: &GopQualityCurve) -> usize {
    let w = gop_damage_weight(curve);
    if w >= 1.0 {
        return cap;
    }
    ((cap as f64 * w) as usize).min(cap)
}

/// Predicted `(clip PSNR, worst-MB PSNR)` of an embed planned as `plan`
/// (bytes per GOP, the apps' `plan_safe_balanced` shape).
///
/// The worst-MB term reads the energy per frame of the WORST stego GOP, not
/// the energy pooled over the stego window (changed 2026-10-26): worst-MB is
/// a maximum, and it sits in the GOP that pays most per bit. On the 2-GOP
/// clips the fit was made on the two coincide within 0.5 dB and every
/// calibration cell keeps its decision; on long clips the pooled figure
/// hides a hot GOP. Measured: a 20-GOP IMG_4138 720p clip (GOPs 15–16 pay
/// 2–4× the energy per bit of the rest) was predicted 18.2 dB worst-MB at
/// 3/16 pooled and measured 9.3–10.5 dB, all in GOPs 15–16; the worst-GOP
/// term predicts 15.8 dB and keeps it at 1/8
/// (`intra-drift-diagnosis.md` follow-up 5).
pub fn predict_quality(curves: &[GopQualityCurve], plan: &[usize], width: u32, height: u32) -> Option<(f64, f64)> {
    let px = width as f64 * height as f64;
    let (mut e_sum, mut frames, mut w_frames, mut w_bits, mut w_dry) = (0.0f64, 0u64, 0u64, 0u64, 0u64);
    let mut e_frame_max = 0.0f64;
    for (c, &bytes) in curves.iter().zip(plan) {
        frames += c.n_frames as u64;
        if bytes == 0 || c.n_dry == 0 {
            continue;
        }
        let a = (bytes as f64 * 8.0) / c.n_dry as f64;
        let e = c.energy_at(a);
        e_sum += e;
        if c.n_frames > 0 {
            e_frame_max = e_frame_max.max(e / c.n_frames as f64);
        }
        w_frames += c.n_frames as u64;
        w_bits += bytes as u64 * 8;
        w_dry += c.n_dry as u64;
    }
    if w_frames == 0 || w_dry == 0 || e_sum <= 0.0 {
        return None;
    }
    let la = det_db(w_bits as f64 / w_dry as f64);
    let x_clip = det_db(e_sum / (frames as f64 * px));
    let x_win = det_db(e_frame_max / px);
    Some((
        CLIP_FIT[0] + CLIP_FIT[1] * x_clip + CLIP_FIT[2] * la,
        WMB_FIT[0] + WMB_FIT[1] * x_win + WMB_FIT[2] * la,
    ))
}

/// The per-clip red end in 1/64ths of α: [`ALPHA64_CAP`] (12) when the embed
/// at 12/64, planned the way the apps plan it (`plan_safe_balanced` over
/// every probed GOP), is predicted to hold both floors with the margins;
/// else [`ALPHA64_FLOOR`] (8). `caps` = the probe's per-GOP tier-0
/// capacities (the planner's samples), `gop_size` = frames per GOP.
pub fn adaptive_max_alpha64(curves: &[GopQualityCurve], caps: &[usize], gop_size: u32, width: u32, height: u32) -> u32 {
    use crate::stego::balanced_allocation::{plan_safe_balanced, PlanOutcome};
    let n_gops = curves.len();
    if n_gops == 0 || caps.len() != n_gops {
        return ALPHA64_FLOOR;
    }
    // The same weighted dry cover the bands are computed over.
    let dry: usize = curves.iter().map(|c| quality_safe_cap(c.n_dry, c)).sum();
    let samples: Vec<(usize, usize)> = caps.iter().copied().enumerate().collect();
    let cal = crate::stego::calibration::AllocationCalibration::H264_1080P_QP26;
    let framed = dry.saturating_mul(ALPHA64_CAP as usize) / 64 / 8;
    let PlanOutcome::Plan { plan, .. } = plan_safe_balanced(&samples, n_gops, framed, gop_size as usize, &cal) else {
        return ALPHA64_FLOOR;
    };
    match predict_quality(curves, &plan, width, height) {
        Some((clip, wmb)) if clip >= CLIP_FLOOR_DB + CLIP_MARGIN_DB && wmb >= WMB_FLOOR_DB + WMB_MARGIN_DB => ALPHA64_CAP,
        _ => ALPHA64_FLOOR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oct8_bin_is_floor_of_eight_log2() {
        for d in [1u128, 2, 3, 5, 7, 100, 1 << 20, (1 << 40) + 12345, u64::MAX as u128 * 977] {
            let exact = (8.0 * (d as f64).log2()).floor() as usize;
            let b = oct8_bin(d);
            // Equal except within float noise of an edge.
            assert!(b == exact || (8.0 * (d as f64).log2() - b as f64).abs() < 1e-6, "d={d} b={b} exact={exact}");
        }
        assert_eq!(oct8_bin(1), 0);
        assert_eq!(oct8_bin(2), 8);
        assert_eq!(oct8_bin(1 << 100), 800);
    }

    #[test]
    fn det_ln_matches_ln() {
        for x in [1e-300, 1e-12, 0.001, 0.5, 0.70710678, 1.0, 1.5, 2.0, 10.0, 12345.678, 1e100] {
            let a = det_ln(x);
            let b = x.ln();
            assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "x={x} {a} vs {b}");
        }
        for y in [0.0, 1e-9, 0.25, 0.5, 1.0] {
            assert!((ln1p_unit(y) - (y as f64).ln_1p()).abs() < 1e-14);
        }
    }

    fn curve(scale: f64) -> GopQualityCurve {
        // E(α) ∝ α^1.4, the corpus shape, scaled.
        let mut c = GopQualityCurve { energy: [0.0; Q_GRID], n_dry: 200_000, n_frames: 30, intra_p_share: [0.0; Q_GRID], intra_p_per_mb: [0.0; Q_GRID] };
        for k in 0..Q_GRID {
            c.energy[k] = scale * (((k + 1) as f64) / 64.0).powf(1.4);
        }
        c
    }

    #[test]
    fn adaptive_max_spans_floor_to_cap() {
        let (w, h) = (1280u32, 720u32);
        let caps = vec![23_000usize, 23_000];
        // Tiny damage → the cap; huge damage → the floor; bad input → the floor.
        let easy = vec![curve(1e3), curve(1e3)];
        let hard = vec![curve(1e12), curve(1e12)];
        assert_eq!(adaptive_max_alpha64(&easy, &caps, 30, w, h), ALPHA64_CAP);
        assert_eq!(adaptive_max_alpha64(&hard, &caps, 30, w, h), ALPHA64_FLOOR);
        assert_eq!(adaptive_max_alpha64(&easy, &caps[..1], 30, w, h), ALPHA64_FLOOR);
        assert_eq!(adaptive_max_alpha64(&[], &[], 30, w, h), ALPHA64_FLOOR);
        // Monotone in the damage scale.
        let mut last = ALPHA64_CAP;
        for s in [1e3, 1e6, 1e7, 3e7, 1e8, 3e8, 1e9, 1e10, 1e12] {
            let k = adaptive_max_alpha64(&[curve(s), curve(s)], &caps, 30, w, h);
            assert!(k <= last && (k == ALPHA64_FLOOR || k == ALPHA64_CAP), "scale {s}: {k} after {last}");
            last = k;
        }
    }

    #[test]
    fn predicted_quality_falls_with_damage_and_alpha() {
        let c = vec![curve(1e8), curve(1e8)];
        let (c1, w1) = predict_quality(&c, &[3000, 3000], 1280, 720).unwrap();
        let (c2, w2) = predict_quality(&c, &[6000, 6000], 1280, 720).unwrap();
        assert!(c2 < c1 && w2 < w1);
        // A plain tail dilutes the clip figure, not the window's worst MB.
        let (c3, w3) = predict_quality(&c, &[6000, 0], 1280, 720).unwrap();
        assert!(c3 > c2);
        assert!((w3 - w2).abs() < 1e-9);
        assert!(predict_quality(&c, &[0, 0], 1280, 720).is_none());
    }

    #[test]
    fn worst_mb_follows_the_hottest_gop_not_the_pooled_window() {
        // One GOP paying 4x the energy per bit: the worst MB is as bad as in a
        // clip where every GOP is that hot; the clip figure is not.
        let mixed = vec![curve(1e8), curve(4e8), curve(1e8), curve(1e8)];
        let hot = vec![curve(4e8), curve(4e8), curve(4e8), curve(4e8)];
        let plan = [3000, 3000, 3000, 3000];
        let (cm, wm) = predict_quality(&mixed, &plan, 1280, 720).unwrap();
        let (ch, wh) = predict_quality(&hot, &plan, 1280, 720).unwrap();
        assert!((wm - wh).abs() < 1e-9);
        assert!(cm > ch);
    }

    fn with_intra_p(per_step: f64) -> GopQualityCurve {
        let mut c = curve(1e8);
        for k in 0..Q_GRID {
            c.intra_p_per_mb[k] = per_step * (k + 1) as f64;
        }
        c
    }

    #[test]
    fn damage_weight_only_touches_intra_flooded_gops() {
        // At or below the bound at α_ref = 8/64: untouched, cap unchanged.
        let calm = with_intra_p(DMG_INTRA_P_PER_MB_MAX / DMG_REF_ALPHA64 as f64);
        assert_eq!(gop_damage_weight(&calm), 1.0);
        assert_eq!(quality_safe_cap(12_345, &calm), 12_345);
        assert_eq!(gop_damage_weight(&curve(1e12)), 1.0); // no intra-P flips at all
        // Linear in α, crossing the bound at α* = 2.5/64: weight 2.5 / 8.
        let hot = with_intra_p(DMG_INTRA_P_PER_MB_MAX / 2.5);
        assert!((gop_damage_weight(&hot) - 2.5 / 8.0).abs() < 1e-12);
        assert!((24_999..=25_000).contains(&quality_safe_cap(80_000, &hot)));
        // Floored, and monotone in the flood intensity.
        assert_eq!(gop_damage_weight(&with_intra_p(100.0 * DMG_INTRA_P_PER_MB_MAX)), DMG_WEIGHT_MIN);
        let mut last = 1.0;
        for s in [0.02, 0.06, 0.12, 0.2, 0.4, 0.8, 2.0, 10.0].map(|x| x * DMG_INTRA_P_PER_MB_MAX) {
            let w = gop_damage_weight(&with_intra_p(s));
            assert!(w <= last && w >= DMG_WEIGHT_MIN && w <= 1.0, "step {s}: {w} after {last}");
            assert!(quality_safe_cap(1000, &with_intra_p(s)) <= 1000);
            last = w;
        }
    }

    #[test]
    fn energy_interpolates_and_is_zero_at_zero() {
        let mut c = GopQualityCurve { energy: [0.0; Q_GRID], n_dry: 1000, n_frames: 30, intra_p_share: [0.0; Q_GRID], intra_p_per_mb: [0.0; Q_GRID] };
        for k in 0..Q_GRID {
            c.energy[k] = (k + 1) as f64 * 10.0;
        }
        assert_eq!(c.energy_at(0.0), 0.0);
        assert!((c.energy_at(1.0 / 128.0) - 5.0).abs() < 1e-12);
        assert!((c.energy_at(8.0 / 64.0) - 80.0).abs() < 1e-9); // grid point α = 8/64
        assert!((c.energy_at(8.5 / 64.0) - 85.0).abs() < 1e-9); // halfway to 9/64
        assert_eq!(c.energy_at(1.0), c.energy[Q_GRID - 1]);
    }
}
