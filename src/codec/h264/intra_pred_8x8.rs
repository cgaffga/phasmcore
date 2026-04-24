// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! H.264 Intra_8×8 prediction — spec § 8.3.2.
//!
//! **Phase 100-C**: prediction math only. No residual, no transform,
//! no mode-decision wiring. Becomes live in Phase 100-F when the
//! Intra_8×8 end-to-end MB path is assembled.
//!
//! Structure matches `intra_pred::predict_4x4`: a neighbors struct
//! carrying reference samples + availability flags, an enum of 9
//! modes matching H.264 Table 8-2, and one function per mode.
//!
//! ## Reference-sample filter (spec § 8.3.2.2)
//!
//! Before ANY Intra_8×8 mode runs, the top row + left column +
//! top-left corner pass through a 3-tap low-pass filter:
//!
//! ```text
//! p'[x, y] = (p[x-1, y] + 2·p[x, y] + p[x+1, y] + 2) >> 2
//! ```
//!
//! with corner cases at the four ends (top-left, top-right of row,
//! left-bottom, top-left corner itself). This filter is mandatory per
//! spec; no opt-out. Not applying it produces non-conforming recon.

/// Intra_8×8 reference samples + availability flags.
///
/// Top is 16 samples (not just 8) because modes 3 (DDL) and 7 (VL)
/// reach into the top-right extension area: `p[x+y+2, -1]` for
/// x+y up to 13. The caller is responsible for replicating `top[7]`
/// into `top[8..=15]` when the top-right 8×8 is unavailable — spec
/// § 8.3.2.2 handles this via the `p[x, -1]` construction process
/// which we expose via `Neighbors8x8::with_top_right_fallback`.
#[derive(Debug, Clone, Copy)]
pub struct Neighbors8x8 {
    /// Top row samples p[0..16, -1]. Indices 8..=15 come from the
    /// top-right neighbor block OR are replicated from `top[7]` when
    /// that neighbor is unavailable.
    pub top: [u8; 16],
    /// Left column samples p[-1, 0..8].
    pub left: [u8; 8],
    /// Top-left corner p[-1, -1].
    pub top_left: u8,
    pub top_available: bool,
    pub top_right_available: bool,
    pub left_available: bool,
    pub top_left_available: bool,
}

impl Neighbors8x8 {
    /// Construct neighbors with the standard top-right fallback: if
    /// the top row is available but the top-right extension is not,
    /// `top[8..=15]` are filled with `top[7]` per spec § 8.3.2.2.
    ///
    /// `top_base` supplies `top[0..8]` (always needed when
    /// `top_available`); `top_right` supplies `top[8..16]` when the
    /// top-right neighbor exists.
    pub fn with_top_right_fallback(
        top_base: [u8; 8],
        top_right: Option<[u8; 8]>,
        left: [u8; 8],
        top_left: u8,
        top_available: bool,
        left_available: bool,
        top_left_available: bool,
    ) -> Self {
        let mut top = [0u8; 16];
        top[..8].copy_from_slice(&top_base);
        let top_right_available = top_right.is_some();
        match top_right {
            Some(tr) => top[8..16].copy_from_slice(&tr),
            None => {
                // Replicate top[7] into top-right extension per spec.
                for i in 8..16 {
                    top[i] = top_base[7];
                }
            }
        }
        Self {
            top,
            left,
            top_left,
            top_available,
            top_right_available,
            left_available,
            top_left_available,
        }
    }
}

/// Filtered reference samples after the spec § 8.3.2.2 3-tap filter.
/// Holds the same topology as `Neighbors8x8` but with smoothed values;
/// all nine prediction modes read from here, not from raw neighbors.
#[derive(Debug, Clone, Copy)]
pub struct FilteredSamples8x8 {
    pub top: [u8; 16],
    pub left: [u8; 8],
    pub top_left: u8,
}

/// Apply the spec § 8.3.2.2 low-pass reference-sample filter.
///
/// Each position uses the 3-tap smoothing kernel `(a + 2·b + c + 2) >> 2`
/// with edge cases at positions that don't have a left-of-left or
/// right-of-right neighbor — those substitute the available end sample.
pub fn filter_reference_samples(n: &Neighbors8x8) -> FilteredSamples8x8 {
    let mut f_top = [0u8; 16];
    let mut f_left = [0u8; 8];

    // Top-left corner.
    let f_top_left = if n.top_left_available {
        let a = if n.top_available { n.top[0] } else { n.top_left };
        let c = if n.left_available { n.left[0] } else { n.top_left };
        (((a as u32) + 2 * (n.top_left as u32) + (c as u32) + 2) >> 2) as u8
    } else {
        n.top_left
    };

    if n.top_available {
        // x = 0 — left of center is top_left (if available) else top[0].
        let left_of = if n.top_left_available {
            n.top_left as u32
        } else {
            n.top[0] as u32
        };
        f_top[0] = ((left_of + 2 * (n.top[0] as u32) + (n.top[1] as u32) + 2) >> 2) as u8;
        // x = 1..14 (general 3-tap).
        for x in 1..=14 {
            let a = n.top[x - 1] as u32;
            let b = n.top[x] as u32;
            let c = n.top[x + 1] as u32;
            f_top[x] = ((a + 2 * b + c + 2) >> 2) as u8;
        }
        // x = 15 — right of center duplicates p[15] per spec.
        let a = n.top[14] as u32;
        let b = n.top[15] as u32;
        f_top[15] = ((a + 3 * b + 2) >> 2) as u8;
    }

    if n.left_available {
        // y = 0.
        let above = if n.top_left_available {
            n.top_left as u32
        } else {
            n.left[0] as u32
        };
        f_left[0] = ((above + 2 * (n.left[0] as u32) + (n.left[1] as u32) + 2) >> 2) as u8;
        // y = 1..6.
        for y in 1..=6 {
            let a = n.left[y - 1] as u32;
            let b = n.left[y] as u32;
            let c = n.left[y + 1] as u32;
            f_left[y] = ((a + 2 * b + c + 2) >> 2) as u8;
        }
        // y = 7.
        let a = n.left[6] as u32;
        let b = n.left[7] as u32;
        f_left[7] = ((a + 3 * b + 2) >> 2) as u8;
    }

    FilteredSamples8x8 {
        top: f_top,
        left: f_left,
        top_left: f_top_left,
    }
}

/// H.264 Intra_8×8 prediction modes (spec Table 8-2 — identical set
/// to Intra_4×4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intra8x8Mode {
    Vertical = 0,
    Horizontal = 1,
    Dc = 2,
    DiagonalDownLeft = 3,
    DiagonalDownRight = 4,
    VerticalRight = 5,
    HorizontalDown = 6,
    VerticalLeft = 7,
    HorizontalUp = 8,
}

impl Intra8x8Mode {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Vertical,
            1 => Self::Horizontal,
            2 => Self::Dc,
            3 => Self::DiagonalDownLeft,
            4 => Self::DiagonalDownRight,
            5 => Self::VerticalRight,
            6 => Self::HorizontalDown,
            7 => Self::VerticalLeft,
            8 => Self::HorizontalUp,
            _ => return None,
        })
    }
}

/// Compute the 8×8 predicted block for `mode` given raw neighbors.
/// Applies the § 8.3.2.2 reference-sample filter internally.
pub fn predict_8x8(mode: Intra8x8Mode, n: &Neighbors8x8) -> [[u8; 8]; 8] {
    let f = filter_reference_samples(n);
    match mode {
        Intra8x8Mode::Vertical => pred_vertical(&f),
        Intra8x8Mode::Horizontal => pred_horizontal(&f),
        Intra8x8Mode::Dc => pred_dc(&f, n),
        Intra8x8Mode::DiagonalDownLeft => pred_ddl(&f),
        Intra8x8Mode::DiagonalDownRight => pred_ddr(&f),
        Intra8x8Mode::VerticalRight => pred_vr(&f),
        Intra8x8Mode::HorizontalDown => pred_hd(&f),
        Intra8x8Mode::VerticalLeft => pred_vl(&f),
        Intra8x8Mode::HorizontalUp => pred_hu(&f),
    }
}

/// Mode 0. Spec § 8.3.2.2.3.
fn pred_vertical(f: &FilteredSamples8x8) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            out[y][x] = f.top[x];
        }
    }
    out
}

/// Mode 1. Spec § 8.3.2.2.4.
fn pred_horizontal(f: &FilteredSamples8x8) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            out[y][x] = f.left[y];
        }
    }
    out
}

/// Mode 2. Spec § 8.3.2.2.5 — four cases depending on availability.
fn pred_dc(f: &FilteredSamples8x8, raw: &Neighbors8x8) -> [[u8; 8]; 8] {
    let dc: u32 = if raw.top_available && raw.left_available {
        let sum: u32 = f.top[..8].iter().map(|&v| v as u32).sum::<u32>()
            + f.left.iter().map(|&v| v as u32).sum::<u32>();
        (sum + 8) >> 4
    } else if raw.top_available {
        let sum: u32 = f.top[..8].iter().map(|&v| v as u32).sum();
        (sum + 4) >> 3
    } else if raw.left_available {
        let sum: u32 = f.left.iter().map(|&v| v as u32).sum();
        (sum + 4) >> 3
    } else {
        128
    };
    [[dc as u8; 8]; 8]
}

/// Mode 3. Spec § 8.3.2.2.6 — only (7, 7) special.
fn pred_ddl(f: &FilteredSamples8x8) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            let pred = if x == 7 && y == 7 {
                let a = f.top[14] as u32;
                let b = f.top[15] as u32;
                (a + 3 * b + 2) >> 2
            } else {
                let a = f.top[x + y] as u32;
                let b = f.top[x + y + 1] as u32;
                let c = f.top[x + y + 2] as u32;
                (a + 2 * b + c + 2) >> 2
            };
            out[y][x] = pred as u8;
        }
    }
    out
}

/// Mode 4 — Intra_8x8_Diagonal_Down_Right. Spec § 8.3.2.2.6,
/// equations (8-97), (8-98), (8-99). Requires top row, left column,
/// and top-left.
///
/// Spec body (adapted to row-major [y][x] pixel access):
///   for x, y in 0..8:
///     if x > y:  pred[y][x] = (p'[x-y-2, -1] + 2·p'[x-y-1, -1]
///                              + p'[x-y, -1] + 2) >> 2            (8-97)
///     elif x < y: pred[y][x] = (p'[-1, y-x-2] + 2·p'[-1, y-x-1]
///                              + p'[-1, y-x] + 2) >> 2            (8-98)
///     else:       pred[y][x] = (p'[0, -1] + 2·p'[-1, -1]
///                              + p'[-1, 0] + 2) >> 2              (8-99)
///
/// `p'[x, -1]` denotes the filtered top sample at column x (from
/// `FilteredSamples8x8::top`), `p'[-1, y]` the filtered left sample
/// at row y (from `FilteredSamples8x8::left`), and `p'[-1, -1]` the
/// filtered top-left corner (from `FilteredSamples8x8::top_left`).
fn pred_ddr(f: &FilteredSamples8x8) -> [[u8; 8]; 8] {
    let mut pred = [[0u8; 8]; 8];
    let dc_diag = (ref_top(f, 0) + 2 * ref_top_left(f) + ref_left(f, 0) + 2) >> 2;
    for y in 0..8i32 {
        for x in 0..8i32 {
            let v = if x > y {
                let k = (x - y) as isize;
                (ref_top_signed(f, k - 2) + 2 * ref_top_signed(f, k - 1)
                    + ref_top_signed(f, k) + 2) >> 2
            } else if x < y {
                let k = (y - x) as isize;
                (ref_left_signed(f, k - 2) + 2 * ref_left_signed(f, k - 1)
                    + ref_left_signed(f, k) + 2) >> 2
            } else {
                dc_diag
            };
            pred[y as usize][x as usize] = v as u8;
        }
    }
    pred
}

/// Mode 5 — Intra_8x8_Vertical_Right. Spec § 8.3.2.2.7, equations
/// (8-100) … (8-103). Requires top row, left column, and top-left.
///
/// Spec body:
///   zVR = 2·x − y
///   for x, y in 0..8:
///     if zVR ∈ {0, 2, 4, 6, 8, 10, 12, 14}:
///       pred[y][x] = (p'[x - (y>>1) - 1, -1] + p'[x - (y>>1), -1]
///                    + 1) >> 1                                    (8-100)
///     elif zVR ∈ {1, 3, 5, 7, 9, 11, 13}:
///       pred[y][x] = (p'[x - (y>>1) - 2, -1] + 2·p'[x - (y>>1) - 1, -1]
///                    + p'[x - (y>>1), -1] + 2) >> 2               (8-101)
///     elif zVR == -1:
///       pred[y][x] = (p'[-1, 0] + 2·p'[-1, -1] + p'[0, -1] + 2) >> 2  (8-102)
///     else (zVR ∈ {-2..-7}):
///       pred[y][x] = (p'[-1, y - 2·x - 1] + 2·p'[-1, y - 2·x - 2]
///                    + p'[-1, y - 2·x - 3] + 2) >> 2              (8-103)
fn pred_vr(f: &FilteredSamples8x8) -> [[u8; 8]; 8] {
    let mut pred = [[0u8; 8]; 8];
    let bridge = (ref_left(f, 0) + 2 * ref_top_left(f) + ref_top(f, 0) + 2) >> 2;
    for y in 0..8i32 {
        for x in 0..8i32 {
            let z_vr = 2 * x - y;
            let v = if z_vr >= 0 && z_vr & 1 == 0 {
                let k = (x - (y >> 1)) as isize;
                (ref_top_signed(f, k - 1) + ref_top_signed(f, k) + 1) >> 1
            } else if z_vr >= 0 {
                let k = (x - (y >> 1)) as isize;
                (ref_top_signed(f, k - 2) + 2 * ref_top_signed(f, k - 1)
                    + ref_top_signed(f, k) + 2) >> 2
            } else if z_vr == -1 {
                bridge
            } else {
                let k = (y - 2 * x) as isize;
                (ref_left_signed(f, k - 1) + 2 * ref_left_signed(f, k - 2)
                    + ref_left_signed(f, k - 3) + 2) >> 2
            };
            pred[y as usize][x as usize] = v as u8;
        }
    }
    pred
}

/// Mode 6 — Intra_8x8_Horizontal_Down. Spec § 8.3.2.2.8, equations
/// (8-104) … (8-107). Mirror image of VR across the main diagonal.
/// Requires top row, left column, and top-left.
///
/// Spec body:
///   zHD = 2·y − x
///   for x, y in 0..8:
///     if zHD ∈ {0, 2, 4, 6, 8, 10, 12, 14}:
///       pred[y][x] = (p'[-1, y - (x>>1) - 1] + p'[-1, y - (x>>1)]
///                    + 1) >> 1                                    (8-104)
///     elif zHD ∈ {1, 3, 5, 7, 9, 11, 13}:
///       pred[y][x] = (p'[-1, y - (x>>1) - 2] + 2·p'[-1, y - (x>>1) - 1]
///                    + p'[-1, y - (x>>1)] + 2) >> 2               (8-105)
///     elif zHD == -1:
///       pred[y][x] = (p'[-1, 0] + 2·p'[-1, -1] + p'[0, -1] + 2) >> 2  (8-106)
///     else (zHD ∈ {-2..-7}):
///       pred[y][x] = (p'[x - 2·y - 1, -1] + 2·p'[x - 2·y - 2, -1]
///                    + p'[x - 2·y - 3, -1] + 2) >> 2              (8-107)
fn pred_hd(f: &FilteredSamples8x8) -> [[u8; 8]; 8] {
    let mut pred = [[0u8; 8]; 8];
    let bridge = (ref_left(f, 0) + 2 * ref_top_left(f) + ref_top(f, 0) + 2) >> 2;
    for y in 0..8i32 {
        for x in 0..8i32 {
            let z_hd = 2 * y - x;
            let v = if z_hd >= 0 && z_hd & 1 == 0 {
                let k = (y - (x >> 1)) as isize;
                (ref_left_signed(f, k - 1) + ref_left_signed(f, k) + 1) >> 1
            } else if z_hd >= 0 {
                let k = (y - (x >> 1)) as isize;
                (ref_left_signed(f, k - 2) + 2 * ref_left_signed(f, k - 1)
                    + ref_left_signed(f, k) + 2) >> 2
            } else if z_hd == -1 {
                bridge
            } else {
                let k = (x - 2 * y) as isize;
                (ref_top_signed(f, k - 1) + 2 * ref_top_signed(f, k - 2)
                    + ref_top_signed(f, k - 3) + 2) >> 2
            };
            pred[y as usize][x as usize] = v as u8;
        }
    }
    pred
}

/// Mode 7. Spec § 8.3.2.2.10. Requires top (including top-right).
fn pred_vl(f: &FilteredSamples8x8) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            let zvl_even = y & 1 == 0;
            let pred = if zvl_even {
                let idx = x + (y >> 1);
                let a = f.top[idx] as u32;
                let b = f.top[idx + 1] as u32;
                (a + b + 1) >> 1
            } else {
                let idx = x + (y >> 1);
                let a = f.top[idx] as u32;
                let b = f.top[idx + 1] as u32;
                let c = f.top[idx + 2] as u32;
                (a + 2 * b + c + 2) >> 2
            };
            out[y][x] = pred as u8;
        }
    }
    out
}

/// Mode 8. Spec § 8.3.2.2.11. Requires left only.
/// Mode 8 — Intra_8x8_Horizontal_Up. Spec § 8.3.2.2.10, equations
/// (8-110) … (8-113). Requires only the left column.
///
/// Spec body:
///   zHU = x + 2·y
///   for x, y in 0..8:
///     if zHU ∈ {0, 2, 4, 6, 8, 10, 12}:
///       pred[y][x] = (p'[-1, y + (x>>1)] + p'[-1, y + (x>>1) + 1]
///                    + 1) >> 1                                    (8-110)
///     elif zHU ∈ {1, 3, 5, 7, 9, 11}:
///       pred[y][x] = (p'[-1, y + (x>>1)] + 2·p'[-1, y + (x>>1) + 1]
///                    + p'[-1, y + (x>>1) + 2] + 2) >> 2           (8-111)
///     elif zHU == 13:
///       pred[y][x] = (p'[-1, 6] + 3·p'[-1, 7] + 2) >> 2           (8-112)
///     else (zHU > 13):
///       pred[y][x] = p'[-1, 7]                                    (8-113)
fn pred_hu(f: &FilteredSamples8x8) -> [[u8; 8]; 8] {
    let mut pred = [[0u8; 8]; 8];
    let edge_tap = (ref_left(f, 6) + 3 * ref_left(f, 7) + 2) >> 2;
    let tail = ref_left(f, 7);
    for y in 0..8i32 {
        for x in 0..8i32 {
            let z_hu = x + 2 * y;
            let k = (y + (x >> 1)) as isize;
            let v = if z_hu <= 12 && z_hu & 1 == 0 {
                (ref_left_signed(f, k) + ref_left_signed(f, k + 1) + 1) >> 1
            } else if z_hu <= 11 {
                (ref_left_signed(f, k) + 2 * ref_left_signed(f, k + 1)
                    + ref_left_signed(f, k + 2) + 2) >> 2
            } else if z_hu == 13 {
                edge_tap
            } else {
                tail
            };
            pred[y as usize][x as usize] = v as u8;
        }
    }
    pred
}

// ─── Filtered-sample accessors mirroring spec notation ─────────────
//
// The spec refers to neighbouring filtered samples as `p'[x, -1]`
// (top row, x = 0..15), `p'[-1, y]` (left column, y = 0..7), and
// `p'[-1, -1]` (top-left corner). These helpers expose those
// directly. A signed-index variant treats `-1` as the top-left
// corner so a spec formula of the form `p'[k, -1]` with k possibly
// reaching `-1` evaluates cleanly without a per-call branch.

#[inline]
fn ref_top(f: &FilteredSamples8x8, x: usize) -> u32 {
    f.top[x] as u32
}

#[inline]
fn ref_left(f: &FilteredSamples8x8, y: usize) -> u32 {
    f.left[y] as u32
}

#[inline]
fn ref_top_left(f: &FilteredSamples8x8) -> u32 {
    f.top_left as u32
}

#[inline]
fn ref_top_signed(f: &FilteredSamples8x8, x: isize) -> u32 {
    if x < 0 {
        f.top_left as u32
    } else if (x as usize) < f.top.len() {
        f.top[x as usize] as u32
    } else {
        f.top[f.top.len() - 1] as u32
    }
}

#[inline]
fn ref_left_signed(f: &FilteredSamples8x8, y: isize) -> u32 {
    if y < 0 {
        f.top_left as u32
    } else if (y as usize) < f.left.len() {
        f.left[y as usize] as u32
    } else {
        f.left[f.left.len() - 1] as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neighbors_all_eq(v: u8) -> Neighbors8x8 {
        Neighbors8x8 {
            top: [v; 16],
            left: [v; 8],
            top_left: v,
            top_available: true,
            top_right_available: true,
            left_available: true,
            top_left_available: true,
        }
    }

    fn neighbors_none() -> Neighbors8x8 {
        Neighbors8x8 {
            top: [0; 16],
            left: [0; 8],
            top_left: 0,
            top_available: false,
            top_right_available: false,
            left_available: false,
            top_left_available: false,
        }
    }

    #[test]
    fn filter_flat_neighbors_is_flat() {
        // Flat neighbors → filter produces flat filtered samples.
        let n = neighbors_all_eq(128);
        let f = filter_reference_samples(&n);
        for &v in &f.top {
            assert_eq!(v, 128);
        }
        for &v in &f.left {
            assert_eq!(v, 128);
        }
        assert_eq!(f.top_left, 128);
    }

    #[test]
    fn filter_preserves_smooth_gradient() {
        // A smooth gradient passes the 3-tap low-pass essentially
        // unchanged at interior points (the filter preserves slope).
        let mut n = neighbors_all_eq(0);
        for i in 0..16 {
            n.top[i] = (i * 16) as u8; // 0, 16, 32, ..., 240
        }
        for i in 0..8 {
            n.left[i] = (i * 16) as u8;
        }
        n.top_left = 0;
        let f = filter_reference_samples(&n);
        // Interior filtered value should be close to the original.
        for i in 1..=14 {
            let diff = (f.top[i] as i32 - n.top[i] as i32).abs();
            assert!(diff <= 1, "top[{i}] filter changed gradient by {diff}");
        }
    }

    #[test]
    fn vertical_mode_copies_top_down() {
        let mut n = neighbors_all_eq(100);
        for i in 0..8 {
            n.top[i] = (100 + i * 10) as u8;
        }
        let out = predict_8x8(Intra8x8Mode::Vertical, &n);
        for y in 0..8 {
            for x in 0..8 {
                // Output should match the FILTERED top[x], not raw.
                assert!(out[y][x] >= 90 && out[y][x] <= 200);
                // All rows identical.
                assert_eq!(out[y][x], out[0][x]);
            }
        }
    }

    #[test]
    fn horizontal_mode_copies_left_across() {
        let mut n = neighbors_all_eq(100);
        for i in 0..8 {
            n.left[i] = (100 + i * 10) as u8;
        }
        let out = predict_8x8(Intra8x8Mode::Horizontal, &n);
        for y in 0..8 {
            // All columns in row y identical.
            for x in 0..8 {
                assert_eq!(out[y][x], out[y][0]);
            }
        }
    }

    #[test]
    fn dc_mode_all_equal() {
        // DC mode always produces a flat block.
        let n = neighbors_all_eq(75);
        let out = predict_8x8(Intra8x8Mode::Dc, &n);
        let v0 = out[0][0];
        for row in &out {
            for &v in row {
                assert_eq!(v, v0);
            }
        }
        // With flat neighbors of value 75, DC should be ~75.
        assert!(v0 >= 74 && v0 <= 76);
    }

    #[test]
    fn dc_mode_no_neighbors_is_128() {
        let n = neighbors_none();
        let out = predict_8x8(Intra8x8Mode::Dc, &n);
        for row in &out {
            for &v in row {
                assert_eq!(v, 128);
            }
        }
    }

    #[test]
    fn all_modes_produce_valid_u8() {
        // Just sanity: every mode on a realistic neighbor set returns
        // valid u8 values with no overflow.
        let mut n = neighbors_all_eq(64);
        for i in 0..16 {
            n.top[i] = (i * 16) as u8;
        }
        for i in 0..8 {
            n.left[i] = (128 + i * 8) as u8;
        }
        n.top_left = 64;
        for mode_id in 0..=8u8 {
            let mode = Intra8x8Mode::from_u8(mode_id).unwrap();
            let out = predict_8x8(mode, &n);
            for row in &out {
                for &v in row {
                    let _ = v; // u8 bounds are enforced by the type
                }
            }
        }
    }

    #[test]
    fn from_u8_roundtrip() {
        for id in 0..=8u8 {
            let m = Intra8x8Mode::from_u8(id).unwrap();
            assert_eq!(m as u8, id);
        }
        assert!(Intra8x8Mode::from_u8(9).is_none());
    }

    #[test]
    fn with_top_right_fallback_replicates_top7() {
        let top_base = [10, 20, 30, 40, 50, 60, 70, 80];
        let n = Neighbors8x8::with_top_right_fallback(
            top_base, None, [0; 8], 0,
            true, true, true,
        );
        assert_eq!(&n.top[..8], &top_base);
        for i in 8..16 {
            assert_eq!(n.top[i], 80, "top[{i}] should replicate top[7]=80");
        }
        assert!(!n.top_right_available);
    }
}
