// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AV1 v2 damage model: D = E · T · X from bitstream syntax only
//! (spec `docs/design/video/stego-cover-order-and-damage-costs.md` §§ 4–5).
//!
//! Every input is something the decoder sees (position, transform, |c|,
//! qindex, frame index), and no layer's flip changes any of them (sign
//! flips keep |c|; golomb tails are wet). So the encoder, the shadow
//! decoder and every platform compute the same D: plain f32 arithmetic in
//! a fixed order, no fused multiply-add, bases built once with `det_math`.
//!
//! - **E** = (2·|c|·qstep)²: the reconstruction-error energy of a sign flip
//!   (the level moves by 2|c|) through an orthonormal inverse transform.
//!   Monotone in |c|, which fixes the old cost's inversion.
//! - **T** = 1 + κ_t · r, r the share of the GOP still to come.
//! - **X** = 1 + α_x · edge · A: `edge` is the energy of the flip's
//!   edge-constant part on the block's right column and bottom row (what
//!   intra prediction carries onward); `A` the number of pixels right of and
//!   below the block, scaled by β_inter outside the key frame. With A in
//!   pixels, X is the flood's energy over the flip's own (α_x ≈ 1).

use std::sync::OnceLock;

use crate::det_math::{det_cos, det_sin};

/// AV1 8-bit AC quantizer step per qindex (`ac_qlookup_Q3`, AV1 spec
/// § 7.12.2; copied from vendor/phasm-rav1e/src/quantize/tables.rs).
const AC_QLOOKUP_Q3: [u16; 256] = [
    4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
    87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
    104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134,
    136, 138, 140, 142, 144, 146, 148, 150, 152, 155, 158, 161, 164, 167, 170, 173,
    176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 207, 211, 215, 219, 223, 227,
    231, 235, 239, 243, 247, 251, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300,
    305, 311, 317, 323, 329, 335, 341, 347, 353, 359, 366, 373, 380, 387, 394, 401,
    408, 416, 424, 432, 440, 448, 456, 465, 474, 483, 492, 501, 510, 520, 530, 540,
    550, 560, 571, 582, 593, 604, 615, 627, 639, 651, 663, 676, 689, 702, 715, 729,
    743, 757, 771, 786, 801, 816, 832, 848, 864, 881, 898, 915, 933, 951, 969, 988,
    1007, 1026, 1046, 1066, 1087, 1108, 1129, 1151, 1173, 1196, 1219, 1243, 1267, 1292, 1317, 1343,
    1369, 1396, 1423, 1451, 1479, 1508, 1537, 1567, 1597, 1628, 1660, 1692, 1725, 1759, 1793, 1828,];

/// AC quantizer step for an 8-bit `qindex`.
pub fn av1_ac_qstep(qindex: u8) -> u16 {
    AC_QLOOKUP_Q3[qindex as usize]
}

/// One-dimensional transform kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tx1d {
    Dct,
    Adst,
    FlipAdst,
    Idtx,
}

/// (vertical, horizontal) kinds of an AV1 `tx_type` (rav1e `VTX_TAB` /
/// `HTX_TAB`, vendor/phasm-rav1e/src/transform/mod.rs). Out-of-range types
/// (WHT, lossless only) count as DCT.
pub fn tx_1d_kinds(tx_type: u8) -> (Tx1d, Tx1d) {
    use Tx1d::*;
    const V: [Tx1d; 16] = [Dct, Adst, Dct, Adst, FlipAdst, Dct, FlipAdst, Adst, FlipAdst, Idtx, Dct, Idtx, Adst, Idtx, FlipAdst, Idtx];
    const H: [Tx1d; 16] = [Dct, Dct, Adst, Adst, Dct, FlipAdst, FlipAdst, FlipAdst, Adst, Idtx, Idtx, Dct, Idtx, Adst, Idtx, FlipAdst];
    match tx_type {
        t @ 0..=15 => (V[t as usize], H[t as usize]),
        _ => (Dct, Dct),
    }
}

/// Unit-norm basis vectors, indexed [kind][log2 size − 2][frequency][n].
type Bases = [[Vec<Vec<f32>>; 5]; 4];

fn bases() -> &'static Bases {
    static B: OnceLock<Bases> = OnceLock::new();
    B.get_or_init(|| {
        let kinds = [Tx1d::Dct, Tx1d::Adst, Tx1d::FlipAdst, Tx1d::Idtx];
        kinds.map(|kind| [2u32, 3, 4, 5, 6].map(|l| (0..1usize << l).map(|k| basis(kind, 1 << l, k)).collect()))
    })
}

fn basis(kind: Tx1d, n: usize, k: usize) -> Vec<f32> {
    let pi = std::f64::consts::PI;
    let raw: Vec<f64> = (0..n)
        .map(|i| match kind {
            Tx1d::Dct => det_cos(pi * (2 * i + 1) as f64 * k as f64 / (2 * n) as f64),
            // AV1's 4-point ADST is the sinpi transform; the longer ones
            // are DST-IV shaped.
            Tx1d::Adst | Tx1d::FlipAdst if n == 4 => det_sin(pi * ((i + 1) * (2 * k + 1)) as f64 / 9.0),
            Tx1d::Adst | Tx1d::FlipAdst => det_sin(pi * ((2 * i + 1) * (2 * k + 1)) as f64 / (4 * n) as f64),
            Tx1d::Idtx => (i == k) as u8 as f64,
        })
        .collect();
    let norm = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut out: Vec<f32> = raw.iter().map(|v| (v / norm) as f32).collect();
    if kind == Tx1d::FlipAdst {
        out.reverse();
    }
    out
}

fn kind_index(kind: Tx1d) -> usize {
    match kind {
        Tx1d::Dct => 0,
        Tx1d::Adst => 1,
        Tx1d::FlipAdst => 2,
        Tx1d::Idtx => 3,
    }
}

/// Energy of the edge-constant part of a unit flip of coefficient
/// (`fx`, `fy`): right column (h[W−1] · mean v)² plus bottom row
/// (v[H−1] · mean h)².
fn edge_energy(tx_type: u8, w_log2: u8, h_log2: u8, fx: usize, fy: usize) -> f32 {
    let (kv, kh) = tx_1d_kinds(tx_type);
    let b = bases();
    let h = &b[kind_index(kh)][(w_log2.clamp(2, 6) - 2) as usize][fx];
    let v = &b[kind_index(kv)][(h_log2.clamp(2, 6) - 2) as usize][fy];
    let mean = |x: &[f32]| x.iter().sum::<f32>() / x.len() as f32;
    let right = h[h.len() - 1] * mean(v);
    let bottom = v[v.len() - 1] * mean(h);
    right * right + bottom * bottom
}

/// Syntax a position's damage depends on. All of it is decoder-visible.
#[derive(Debug, Clone, Copy, Default)]
pub struct DamageInputs {
    pub plane: u8,
    pub px_x: u16,
    pub px_y: u16,
    pub plane_w: u16,
    pub plane_h: u16,
    pub tx_width_log2: u8,
    pub tx_height_log2: u8,
    pub tx_type: u8,
    pub scan_pos: u16,
    pub abs_c: u16,
    pub qindex: u8,
    pub frame_idx: u16,
    pub gop_len: u16,
    pub is_key_frame: bool,
}

/// Calibration knobs (spec § 13).
#[derive(Debug, Clone, Copy)]
pub struct DamageParams {
    pub kappa_t: f32,
    pub alpha_x: f32,
    pub beta_inter: f32,
}

impl Default for DamageParams {
    fn default() -> Self {
        // alpha_x weighs the edge flood in pixels (energy-consistent at
        // 1.0); calibration in docs/design/video/av1/v2-gate-results.md.
        Self { kappa_t: 0.5, alpha_x: 1.0, beta_inter: 0.1 }
    }
}

impl DamageParams {
    /// Defaults, each overridable by an env var for calibration sweeps
    /// (`PHASM_AV1_KAPPA_T`, `PHASM_AV1_ALPHA_X`, `PHASM_AV1_BETA_INTER`).
    pub fn from_env() -> Self {
        static P: OnceLock<DamageParams> = OnceLock::new();
        *P.get_or_init(|| {
            let d = Self::default();
            let f = |k: &str, v: f32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(v);
            Self {
                kappa_t: f("PHASM_AV1_KAPPA_T", d.kappa_t),
                alpha_x: f("PHASM_AV1_ALPHA_X", d.alpha_x),
                beta_inter: f("PHASM_AV1_BETA_INTER", d.beta_inter),
            }
        })
    }
}

/// D = E · T · X for one position (see the module docs).
pub fn damage(inp: &DamageInputs, p: &DamageParams) -> f32 {
    let qstep = av1_ac_qstep(inp.qindex) as f32;
    let level = 2.0 * inp.abs_c as f32 * qstep;
    let e = level * level;

    let r = if inp.gop_len > 1 {
        (inp.gop_len - 1).saturating_sub(inp.frame_idx) as f32 / (inp.gop_len - 1) as f32
    } else {
        0.0
    };
    let t = 1.0 + p.kappa_t * r;

    let tx_w = 1usize << inp.tx_width_log2.clamp(2, 6);
    let tx_h = 1usize << inp.tx_height_log2.clamp(2, 6);
    let pos = inp.scan_pos as usize;
    let (fx, fy) = ((pos % tx_w).min(tx_w - 1), (pos / tx_w).min(tx_h - 1));
    let edge = edge_energy(inp.tx_type, inp.tx_width_log2, inp.tx_height_log2, fx, fy);
    // Pixels right of and below the block: what intra prediction can carry
    // a constant edge offset over. In pixels, not as a share of the plane:
    // the flood's energy is edge · E · that many pixels, against E for the
    // flip's own block, so alpha_x ≈ 1 is the energy-consistent weight.
    let right = (inp.plane_w as f32 - inp.px_x as f32).max(0.0);
    let below = (inp.plane_h as f32 - inp.px_y as f32).max(0.0);
    let mut area = right * below;
    if !inp.is_key_frame {
        area *= p.beta_inter;
    }
    let x = 1.0 + p.alpha_x * edge * area;

    e * t * x
}

/// Number of shadow damage buckets.
pub const SHADOW_BUCKETS: u8 = 4;

/// Shadow eligibility and damage bucket (spec § 6), from syntax alone so
/// the encoder and every shadow decoder agree. `None`: never a shadow slot
/// (golomb tail, |c| = 0 or ≥ 6, scan_pos 0, or damage above the cap).
/// Otherwise the bucket 0..`SHADOW_BUCKETS`, low damage first: D relative
/// to a |c| = 1 flip's bare energy (2·qstep)², against powers of four.
/// Plain comparisons, no `log2`, so the bucket is bit-stable everywhere.
pub fn shadow_bucket(inp: &DamageInputs, is_golomb_tail: bool, p: &DamageParams) -> Option<u8> {
    if is_golomb_tail || inp.abs_c == 0 || inp.abs_c >= 6 || inp.scan_pos == 0 {
        return None;
    }
    let unit = 2.0 * av1_ac_qstep(inp.qindex) as f32;
    let ratio = damage(inp, p) / (unit * unit);
    match ratio {
        r if r > shadow_ratio_max() => None,
        r if r < 4.0 => Some(0),
        r if r < 16.0 => Some(1),
        r if r < 64.0 => Some(2),
        _ => Some(3),
    }
}

/// Damage cap for shadow slots, relative to a |c| = 1 flip's bare energy.
pub const SHADOW_RATIO_MAX: f32 = 256.0;

/// [`SHADOW_RATIO_MAX`], overridable by `PHASM_AV1_SHADOW_RATIO_MAX` for
/// calibration. The encoder and the decoder must agree on it.
fn shadow_ratio_max() -> f32 {
    static V: OnceLock<f32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PHASM_AV1_SHADOW_RATIO_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(SHADOW_RATIO_MAX)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(abs_c: u16) -> DamageInputs {
        DamageInputs {
            plane: 0,
            px_x: 64,
            px_y: 32,
            plane_w: 640,
            plane_h: 360,
            tx_width_log2: 3,
            tx_height_log2: 3,
            tx_type: 0,
            scan_pos: 1,
            abs_c,
            qindex: 100,
            frame_idx: 3,
            gop_len: 30,
            is_key_frame: true,
        }
    }

    #[test]
    fn ac_qstep_matches_the_av1_table() {
        assert_eq!(
            [0u8, 1, 100, 160, 255].map(av1_ac_qstep),
            [4, 8, 112, 305, 1828]
        );
    }

    #[test]
    fn tx_1d_kinds_match_rav1e() {
        use Tx1d::*;
        assert_eq!(tx_1d_kinds(0), (Dct, Dct));
        assert_eq!(tx_1d_kinds(1), (Adst, Dct));
        assert_eq!(tx_1d_kinds(2), (Dct, Adst));
        assert_eq!(tx_1d_kinds(9), (Idtx, Idtx));
        assert_eq!(tx_1d_kinds(10), (Dct, Idtx));
        assert_eq!(tx_1d_kinds(11), (Idtx, Dct));
        assert_eq!(tx_1d_kinds(15), (Idtx, FlipAdst));
    }

    /// Only vertical-frequency-0 DCT coefficients give a constant right
    /// column (and horizontal-frequency-0 a constant bottom row); the other
    /// AC bases have zero-mean edges.
    #[test]
    fn dc_column_is_edge_constant_and_ac_is_not() {
        let col0 = edge_energy(0, 3, 3, 1, 0); // fx 1, fy 0
        let ac = edge_energy(0, 3, 3, 1, 1);
        assert!(col0 > 0.01, "{col0}");
        assert!(ac < col0 * 1e-6, "{ac} vs {col0}");
    }

    #[test]
    fn damage_is_monotone_in_abs_c() {
        let p = DamageParams::default();
        let d: Vec<f32> = (1..=5).map(|c| damage(&at(c), &p)).collect();
        assert!(d.windows(2).all(|w| w[0] < w[1]), "{d:?}");
    }

    #[test]
    fn temporal_factor_handles_a_one_frame_gop() {
        let p = DamageParams::default();
        let one = DamageInputs { gop_len: 1, frame_idx: 0, ..at(1) };
        let d = damage(&one, &p);
        assert!(d.is_finite() && d > 0.0, "{d}");
        let first = damage(&DamageInputs { frame_idx: 0, ..at(1) }, &p);
        let last = damage(&DamageInputs { frame_idx: 29, ..at(1) }, &p);
        assert!(first > last);
    }

    #[test]
    fn damage_is_finite_and_positive_for_every_valid_tx() {
        let p = DamageParams::default();
        for wl in 2u8..=6 {
            for hl in 2u8..=6 {
                if wl.abs_diff(hl) > 2 {
                    continue; // AV1 aspect ratios go up to 4:1
                }
                let (w, h) = (1usize << wl, 1usize << hl);
                let (cw, ch) = (w.min(32), h.min(32)); // 64-point: 32 coded
                for t in 0..=15u8 {
                    for fy in 0..ch {
                        for fx in 0..cw {
                            let inp = DamageInputs {
                                tx_width_log2: wl,
                                tx_height_log2: hl,
                                tx_type: t,
                                scan_pos: (fy * w + fx) as u16,
                                ..at(1)
                            };
                            let d = damage(&inp, &p);
                            assert!(d.is_finite() && d > 0.0, "{w}x{h} t{t} ({fx},{fy}): {d}");
                        }
                    }
                }
            }
        }
    }

    /// Cross-platform pin: shadow buckets are computed independently by
    /// encoder and decoder, possibly on different CPUs, so D must be
    /// bit-identical everywhere. Re-record only with a written reason.
    #[test]
    fn damage_values_are_pinned() {
        let p = DamageParams::default();
        let cases = [
            at(1),
            at(5),
            DamageInputs { plane: 1, plane_w: 320, plane_h: 180, px_x: 0, px_y: 0, ..at(2) },
            DamageInputs { is_key_frame: false, frame_idx: 12, ..at(3) },
            DamageInputs { tx_type: 3, scan_pos: 8, ..at(1) },
            DamageInputs { tx_type: 6, scan_pos: 2, ..at(2) },
            DamageInputs { tx_type: 9, scan_pos: 9, ..at(1) },
            DamageInputs { tx_type: 15, scan_pos: 4, ..at(4) },
            DamageInputs { tx_width_log2: 2, tx_height_log2: 2, scan_pos: 4, ..at(1) },
            DamageInputs { tx_width_log2: 5, tx_height_log2: 5, scan_pos: 1, qindex: 200, ..at(1) },
            DamageInputs { tx_width_log2: 2, tx_height_log2: 4, tx_type: 1, scan_pos: 3, ..at(2) },
            DamageInputs { gop_len: 1, frame_idx: 0, px_x: 600, px_y: 340, ..at(1) },
        ];
        let got: Vec<u32> = cases.iter().map(|c| damage(c, &p).to_bits()).collect();
        assert_eq!(got, PINNED, "D changed: {:?}", cases.iter().map(|c| damage(c, &p)).collect::<Vec<_>>());
    }

    #[test]
    fn shadow_bucket_excludes_wet_classes() {
        let p = DamageParams::default();
        assert_eq!(shadow_bucket(&at(1), true, &p), None, "golomb tail");
        assert_eq!(shadow_bucket(&at(6), false, &p), None, "reject band");
        assert_eq!(shadow_bucket(&DamageInputs { scan_pos: 0, ..at(1) }, false, &p), None, "DC");
        let low = DamageInputs { px_x: 600, px_y: 340, frame_idx: 29, is_key_frame: false, ..at(1) };
        assert!(shadow_bucket(&low, false, &p).is_some());
    }

    #[test]
    fn shadow_bucket_rises_with_damage() {
        let p = DamageParams::default();
        let low = DamageInputs { px_x: 600, px_y: 340, frame_idx: 29, is_key_frame: false, ..at(1) };
        let b_low = shadow_bucket(&low, false, &p).unwrap();
        let b_high = shadow_bucket(&at(3), false, &p);
        assert_eq!(b_low, 0);
        assert!(b_high.is_none_or(|b| b > b_low), "{b_high:?}");
    }

    // Re-recorded 2026-10-25: X's area in pixels with alpha_x 1 (the
    // energy-consistent weight); matched on aarch64 and x86_64.
    const PINNED: [u32; 12] = [
        1304744985, 1343866291, 1307582864, 1302227608, 1303062883, 1301563702,
        1200483928, 1307904987, 1320067326, 1313876641, 1296217617, 1234791432,
    ];
}
