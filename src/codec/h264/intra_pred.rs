// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 intra prediction — Intra_4x4 (9 modes) and Intra_16x16 (4 modes).
//!
//! Implements the sample prediction step for intra-coded I-frame macroblocks
//! from H.264 spec Section 8.3. Only spatial Y-plane prediction is covered;
//! chroma prediction is deferred (Phase 1b UNIWARD runs on luma only).
//!
//! The entry points are [`predict_4x4`] and [`predict_16x16`]: each takes a
//! [`Neighbors`] snapshot of already-decoded samples from the adjacent
//! blocks and returns a 4×4 or 16×16 predicted sample grid.

/// Neighbor samples required for Intra_4x4 prediction of one block.
///
/// A fully-available 4×4 intra block has:
/// - `top`          : 8 samples (4 directly above + 4 above-right)
/// - `left`         : 4 samples (directly to the left)
/// - `top_left`     : 1 sample (the corner)
///
/// For edge blocks some of these may not be available; the booleans record
/// which parts are valid. When the spec permits it, mode-specific code can
/// substitute a default (128) or replicate a neighboring sample.
#[derive(Debug, Clone, Copy)]
pub struct Neighbors4x4 {
    pub top: [u8; 8],
    pub left: [u8; 4],
    pub top_left: u8,
    pub top_available: bool,
    pub top_right_available: bool,
    pub left_available: bool,
    pub top_left_available: bool,
}

/// Neighbor samples required for Intra_16x16 prediction of a macroblock.
#[derive(Debug, Clone, Copy)]
pub struct Neighbors16x16 {
    pub top: [u8; 16],
    pub left: [u8; 16],
    pub top_left: u8,
    pub top_available: bool,
    pub left_available: bool,
    pub top_left_available: bool,
}

/// Intra_4x4 prediction mode numbers from H.264 Table 8-2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intra4x4Mode {
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

impl Intra4x4Mode {
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

/// Intra_16x16 prediction mode numbers from H.264 Table 8-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intra16x16Mode {
    Vertical = 0,
    Horizontal = 1,
    Dc = 2,
    Plane = 3,
}

impl Intra16x16Mode {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Vertical,
            1 => Self::Horizontal,
            2 => Self::Dc,
            3 => Self::Plane,
            _ => return None,
        })
    }
}

/// Clip an i32 sample to the 0..=255 u8 range.
#[inline]
fn clip1(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Predict one 4×4 block using the specified intra mode.
///
/// Returns a 4×4 predicted sample grid `p[y][x]` where `(y,x)` is row/col
/// inside the block (both 0..=3).
pub fn predict_4x4(mode: Intra4x4Mode, n: &Neighbors4x4) -> [[u8; 4]; 4] {
    match mode {
        Intra4x4Mode::Vertical => pred4x4_vertical(n),
        Intra4x4Mode::Horizontal => pred4x4_horizontal(n),
        Intra4x4Mode::Dc => pred4x4_dc(n),
        Intra4x4Mode::DiagonalDownLeft => pred4x4_diagonal_down_left(n),
        Intra4x4Mode::DiagonalDownRight => pred4x4_diagonal_down_right(n),
        Intra4x4Mode::VerticalRight => pred4x4_vertical_right(n),
        Intra4x4Mode::HorizontalDown => pred4x4_horizontal_down(n),
        Intra4x4Mode::VerticalLeft => pred4x4_vertical_left(n),
        Intra4x4Mode::HorizontalUp => pred4x4_horizontal_up(n),
    }
}

// -----------------------------------------------------------------------
// Intra_4x4 modes — H.264 spec Section 8.3.1
// -----------------------------------------------------------------------

fn pred4x4_vertical(n: &Neighbors4x4) -> [[u8; 4]; 4] {
    // Mode 0: every row = top[0..=3]. Requires `top_available`.
    let mut p = [[0u8; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            p[y][x] = if n.top_available { n.top[x] } else { 128 };
        }
    }
    p
}

fn pred4x4_horizontal(n: &Neighbors4x4) -> [[u8; 4]; 4] {
    // Mode 1: every column = left[0..=3]. Requires `left_available`.
    let mut p = [[0u8; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            p[y][x] = if n.left_available { n.left[y] } else { 128 };
        }
    }
    p
}

fn pred4x4_dc(n: &Neighbors4x4) -> [[u8; 4]; 4] {
    // Mode 2: DC. Average of available top + left samples, rounded.
    let dc = match (n.top_available, n.left_available) {
        (true, true) => {
            let s: u32 = n.top[..4].iter().map(|&v| v as u32).sum::<u32>()
                + n.left.iter().map(|&v| v as u32).sum::<u32>();
            ((s + 4) >> 3) as u8
        }
        (true, false) => {
            let s: u32 = n.top[..4].iter().map(|&v| v as u32).sum();
            ((s + 2) >> 2) as u8
        }
        (false, true) => {
            let s: u32 = n.left.iter().map(|&v| v as u32).sum();
            ((s + 2) >> 2) as u8
        }
        (false, false) => 128,
    };
    [[dc; 4]; 4]
}

fn pred4x4_diagonal_down_left(n: &Neighbors4x4) -> [[u8; 4]; 4] {
    // Mode 3: diagonal down-left. Requires `top_available`. When
    // `top_right_available == false`, samples top[4..=7] are replaced by top[3].
    let mut s = [0u16; 8];
    for i in 0..8 {
        s[i] = (if i < 4 || n.top_right_available { n.top[i] } else { n.top[3] }) as u16;
    }

    let mut p = [[0u8; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            let v = if x == 3 && y == 3 {
                (s[6] + 3 * s[7] + 2) >> 2
            } else {
                (s[x + y] + 2 * s[x + y + 1] + s[x + y + 2] + 2) >> 2
            };
            p[y][x] = v as u8;
        }
    }
    p
}

fn pred4x4_diagonal_down_right(n: &Neighbors4x4) -> [[u8; 4]; 4] {
    // Mode 4: diagonal down-right. Requires top, left, and top-left.
    // Build reference strip s[] of 9 samples: left[3..=0], top_left, top[0..=3].
    let mut s = [0i32; 9];
    s[0] = n.left[3] as i32;
    s[1] = n.left[2] as i32;
    s[2] = n.left[1] as i32;
    s[3] = n.left[0] as i32;
    s[4] = n.top_left as i32;
    s[5] = n.top[0] as i32;
    s[6] = n.top[1] as i32;
    s[7] = n.top[2] as i32;
    s[8] = n.top[3] as i32;

    let mut p = [[0u8; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            // zHD = x - y; reference index = 4 + (x - y)
            let idx = (4 + x as i32 - y as i32) as usize;
            // Use the standard three-tap filter (s[idx-1] + 2*s[idx] + s[idx+1] + 2) >> 2.
            let v = (s[idx - 1] + 2 * s[idx] + s[idx + 1] + 2) >> 2;
            p[y][x] = clip1(v);
        }
    }
    p
}

fn pred4x4_vertical_right(n: &Neighbors4x4) -> [[u8; 4]; 4] {
    // Mode 5: vertical-right. Requires top, top_left, left.
    let tl = n.top_left as i32;
    let t0 = n.top[0] as i32;
    let t1 = n.top[1] as i32;
    let t2 = n.top[2] as i32;
    let t3 = n.top[3] as i32;
    let l0 = n.left[0] as i32;
    let l1 = n.left[1] as i32;
    let l2 = n.left[2] as i32;

    let mut p = [[0i32; 4]; 4];
    // Spec 8.3.1.2.5: zVR = 2*x - y
    // Piecewise formulas per zVR value (see spec Table 8-7).
    p[0][0] = (tl + t0 + 1) >> 1;
    p[0][1] = (t0 + t1 + 1) >> 1;
    p[0][2] = (t1 + t2 + 1) >> 1;
    p[0][3] = (t2 + t3 + 1) >> 1;
    p[1][0] = (l0 + 2 * tl + t0 + 2) >> 2;
    p[1][1] = (tl + 2 * t0 + t1 + 2) >> 2;
    p[1][2] = (t0 + 2 * t1 + t2 + 2) >> 2;
    p[1][3] = (t1 + 2 * t2 + t3 + 2) >> 2;
    p[2][0] = (tl + 2 * l0 + l1 + 2) >> 2;
    p[2][1] = p[0][0];
    p[2][2] = p[0][1];
    p[2][3] = p[0][2];
    p[3][0] = (l0 + 2 * l1 + l2 + 2) >> 2;
    p[3][1] = p[1][0];
    p[3][2] = p[1][1];
    p[3][3] = p[1][2];

    let mut out = [[0u8; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            out[y][x] = clip1(p[y][x]);
        }
    }
    out
}

fn pred4x4_horizontal_down(n: &Neighbors4x4) -> [[u8; 4]; 4] {
    // Mode 6: horizontal-down. Requires top, top_left, left.
    let tl = n.top_left as i32;
    let t0 = n.top[0] as i32;
    let t1 = n.top[1] as i32;
    let t2 = n.top[2] as i32;
    let l0 = n.left[0] as i32;
    let l1 = n.left[1] as i32;
    let l2 = n.left[2] as i32;
    let l3 = n.left[3] as i32;

    let mut p = [[0i32; 4]; 4];
    p[0][0] = (tl + l0 + 1) >> 1;
    p[0][1] = (l0 + 2 * tl + t0 + 2) >> 2;
    p[0][2] = (tl + 2 * t0 + t1 + 2) >> 2;
    p[0][3] = (t0 + 2 * t1 + t2 + 2) >> 2;
    p[1][0] = (l0 + l1 + 1) >> 1;
    p[1][1] = (tl + 2 * l0 + l1 + 2) >> 2;
    p[1][2] = p[0][0];
    p[1][3] = p[0][1];
    p[2][0] = (l1 + l2 + 1) >> 1;
    p[2][1] = (l0 + 2 * l1 + l2 + 2) >> 2;
    p[2][2] = p[1][0];
    p[2][3] = p[1][1];
    p[3][0] = (l2 + l3 + 1) >> 1;
    p[3][1] = (l1 + 2 * l2 + l3 + 2) >> 2;
    p[3][2] = p[2][0];
    p[3][3] = p[2][1];

    let mut out = [[0u8; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            out[y][x] = clip1(p[y][x]);
        }
    }
    out
}

fn pred4x4_vertical_left(n: &Neighbors4x4) -> [[u8; 4]; 4] {
    // Mode 7: vertical-left. Requires top; when top_right_available is false,
    // replicate top[3] into top[4..=7].
    let mut s = [0i32; 8];
    for i in 0..8 {
        s[i] = (if i < 4 || n.top_right_available { n.top[i] } else { n.top[3] }) as i32;
    }

    let mut p = [[0i32; 4]; 4];
    p[0][0] = (s[0] + s[1] + 1) >> 1;
    p[0][1] = (s[1] + s[2] + 1) >> 1;
    p[0][2] = (s[2] + s[3] + 1) >> 1;
    p[0][3] = (s[3] + s[4] + 1) >> 1;
    p[1][0] = (s[0] + 2 * s[1] + s[2] + 2) >> 2;
    p[1][1] = (s[1] + 2 * s[2] + s[3] + 2) >> 2;
    p[1][2] = (s[2] + 2 * s[3] + s[4] + 2) >> 2;
    p[1][3] = (s[3] + 2 * s[4] + s[5] + 2) >> 2;
    p[2][0] = (s[1] + s[2] + 1) >> 1;
    p[2][1] = (s[2] + s[3] + 1) >> 1;
    p[2][2] = (s[3] + s[4] + 1) >> 1;
    p[2][3] = (s[4] + s[5] + 1) >> 1;
    p[3][0] = (s[1] + 2 * s[2] + s[3] + 2) >> 2;
    p[3][1] = (s[2] + 2 * s[3] + s[4] + 2) >> 2;
    p[3][2] = (s[3] + 2 * s[4] + s[5] + 2) >> 2;
    p[3][3] = (s[4] + 2 * s[5] + s[6] + 2) >> 2;

    let mut out = [[0u8; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            out[y][x] = clip1(p[y][x]);
        }
    }
    out
}

fn pred4x4_horizontal_up(n: &Neighbors4x4) -> [[u8; 4]; 4] {
    // Mode 8: horizontal-up. Requires left.
    let l0 = n.left[0] as i32;
    let l1 = n.left[1] as i32;
    let l2 = n.left[2] as i32;
    let l3 = n.left[3] as i32;

    let mut p = [[0i32; 4]; 4];
    p[0][0] = (l0 + l1 + 1) >> 1;
    p[0][1] = (l0 + 2 * l1 + l2 + 2) >> 2;
    p[0][2] = (l1 + l2 + 1) >> 1;
    p[0][3] = (l1 + 2 * l2 + l3 + 2) >> 2;
    p[1][0] = p[0][2];
    p[1][1] = p[0][3];
    p[1][2] = (l2 + l3 + 1) >> 1;
    p[1][3] = (l2 + 3 * l3 + 2) >> 2;
    p[2][0] = p[1][2];
    p[2][1] = p[1][3];
    p[2][2] = l3;
    p[2][3] = l3;
    p[3][0] = l3;
    p[3][1] = l3;
    p[3][2] = l3;
    p[3][3] = l3;

    let mut out = [[0u8; 4]; 4];
    for y in 0..4 {
        for x in 0..4 {
            out[y][x] = clip1(p[y][x]);
        }
    }
    out
}

// -----------------------------------------------------------------------
// Intra Chroma 8×8 modes — H.264 spec Section 8.3.4 (4:2:0)
// -----------------------------------------------------------------------

/// Neighbor samples required for chroma 8×8 intra prediction.
#[derive(Debug, Clone, Copy)]
pub struct NeighborsChroma8x8 {
    pub top: [u8; 8],
    pub left: [u8; 8],
    pub top_left: u8,
    pub top_available: bool,
    pub left_available: bool,
    pub top_left_available: bool,
}

/// Chroma intra 8×8 prediction mode numbers from H.264 spec 8.3.4.
/// The ordering matches the ue(v)-encoded `intra_chroma_pred_mode` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraChroma8x8Mode {
    Dc = 0,
    Horizontal = 1,
    Vertical = 2,
    Plane = 3,
}

impl IntraChroma8x8Mode {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Dc,
            1 => Self::Horizontal,
            2 => Self::Vertical,
            3 => Self::Plane,
            _ => return None,
        })
    }
}

/// Predict an 8×8 chroma block using the specified intra mode.
pub fn predict_chroma_8x8(mode: IntraChroma8x8Mode, n: &NeighborsChroma8x8) -> [[u8; 8]; 8] {
    match mode {
        IntraChroma8x8Mode::Dc => pred_chroma_dc(n),
        IntraChroma8x8Mode::Horizontal => pred_chroma_horizontal(n),
        IntraChroma8x8Mode::Vertical => pred_chroma_vertical(n),
        IntraChroma8x8Mode::Plane => pred_chroma_plane(n),
    }
}

fn pred_chroma_vertical(n: &NeighborsChroma8x8) -> [[u8; 8]; 8] {
    let mut p = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            p[y][x] = if n.top_available { n.top[x] } else { 128 };
        }
    }
    p
}

fn pred_chroma_horizontal(n: &NeighborsChroma8x8) -> [[u8; 8]; 8] {
    let mut p = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            p[y][x] = if n.left_available { n.left[y] } else { 128 };
        }
    }
    p
}

/// Chroma DC prediction — spec Section 8.3.4.2. Unlike Intra_16x16 DC, the
/// 8×8 chroma block is split into four 4×4 quadrants, each with its own DC
/// value: the top-left and bottom-right use top+left neighbours, while
/// top-right and bottom-left use only one side when both exist.
fn pred_chroma_dc(n: &NeighborsChroma8x8) -> [[u8; 8]; 8] {
    // Accumulate partial sums per half.
    let top_lo: u32 = if n.top_available {
        n.top[..4].iter().map(|&v| v as u32).sum()
    } else {
        0
    };
    let top_hi: u32 = if n.top_available {
        n.top[4..].iter().map(|&v| v as u32).sum()
    } else {
        0
    };
    let left_lo: u32 = if n.left_available {
        n.left[..4].iter().map(|&v| v as u32).sum()
    } else {
        0
    };
    let left_hi: u32 = if n.left_available {
        n.left[4..].iter().map(|&v| v as u32).sum()
    } else {
        0
    };

    let (dc_tl, dc_tr, dc_bl, dc_br) = match (n.top_available, n.left_available) {
        (true, true) => (
            ((top_lo + left_lo + 4) >> 3) as u8,
            ((top_hi + 2) >> 2) as u8,
            ((left_hi + 2) >> 2) as u8,
            ((top_hi + left_hi + 4) >> 3) as u8,
        ),
        (true, false) => {
            let a = ((top_lo + 2) >> 2) as u8;
            let b = ((top_hi + 2) >> 2) as u8;
            (a, b, a, b)
        }
        (false, true) => {
            let a = ((left_lo + 2) >> 2) as u8;
            let b = ((left_hi + 2) >> 2) as u8;
            (a, a, b, b)
        }
        (false, false) => (128u8, 128u8, 128u8, 128u8),
    };

    let mut p = [[0u8; 8]; 8];
    for y in 0..4 {
        for x in 0..4 {
            p[y][x] = dc_tl;
        }
        for x in 4..8 {
            p[y][x] = dc_tr;
        }
    }
    for y in 4..8 {
        for x in 0..4 {
            p[y][x] = dc_bl;
        }
        for x in 4..8 {
            p[y][x] = dc_br;
        }
    }
    p
}

/// Chroma plane prediction — spec Section 8.3.4.5. Same shape as Intra_16x16
/// plane but with an 8×8 block and k = 1..=4 instead of 1..=8.
fn pred_chroma_plane(n: &NeighborsChroma8x8) -> [[u8; 8]; 8] {
    let top = &n.top;
    let left = &n.left;
    let tl = n.top_left as i32;

    // H = sum_{i=0..=3} (i+1) * (top[4+i] - top[2-i]), where top[-1] = tl
    let mut h = 0i32;
    for i in 0..4 {
        let t_hi = top[4 + i] as i32;
        let t_lo = if 2 >= i { top[2 - i] as i32 } else { tl };
        h += (i as i32 + 1) * (t_hi - t_lo);
    }
    // V = sum_{j=0..=3} (j+1) * (left[4+j] - left[2-j])
    let mut v = 0i32;
    for j in 0..4 {
        let l_hi = left[4 + j] as i32;
        let l_lo = if 2 >= j { left[2 - j] as i32 } else { tl };
        v += (j as i32 + 1) * (l_hi - l_lo);
    }

    // Spec scales for 8×8 plane (different from 16×16): b, c = (34*H + 32) >> 6
    let b = (34 * h + 32) >> 6;
    let c = (34 * v + 32) >> 6;
    let a = 16 * (top[7] as i32 + left[7] as i32);

    let mut p = [[0u8; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            let val = (a + b * (x as i32 - 3) + c * (y as i32 - 3) + 16) >> 5;
            p[y][x] = clip1(val);
        }
    }
    p
}

// -----------------------------------------------------------------------
// Intra_16x16 modes — H.264 spec Section 8.3.3
// -----------------------------------------------------------------------

/// Predict a 16×16 luma block using the specified Intra_16x16 mode.
pub fn predict_16x16(mode: Intra16x16Mode, n: &Neighbors16x16) -> [[u8; 16]; 16] {
    match mode {
        Intra16x16Mode::Vertical => pred16x16_vertical(n),
        Intra16x16Mode::Horizontal => pred16x16_horizontal(n),
        Intra16x16Mode::Dc => pred16x16_dc(n),
        Intra16x16Mode::Plane => pred16x16_plane(n),
    }
}

fn pred16x16_vertical(n: &Neighbors16x16) -> [[u8; 16]; 16] {
    let mut p = [[0u8; 16]; 16];
    for y in 0..16 {
        for x in 0..16 {
            p[y][x] = if n.top_available { n.top[x] } else { 128 };
        }
    }
    p
}

fn pred16x16_horizontal(n: &Neighbors16x16) -> [[u8; 16]; 16] {
    let mut p = [[0u8; 16]; 16];
    for y in 0..16 {
        for x in 0..16 {
            p[y][x] = if n.left_available { n.left[y] } else { 128 };
        }
    }
    p
}

fn pred16x16_dc(n: &Neighbors16x16) -> [[u8; 16]; 16] {
    let dc = match (n.top_available, n.left_available) {
        (true, true) => {
            let sum: u32 = n.top.iter().map(|&v| v as u32).sum::<u32>()
                + n.left.iter().map(|&v| v as u32).sum::<u32>();
            ((sum + 16) >> 5) as u8
        }
        (true, false) => {
            let sum: u32 = n.top.iter().map(|&v| v as u32).sum();
            ((sum + 8) >> 4) as u8
        }
        (false, true) => {
            let sum: u32 = n.left.iter().map(|&v| v as u32).sum();
            ((sum + 8) >> 4) as u8
        }
        (false, false) => 128,
    };
    [[dc; 16]; 16]
}

fn pred16x16_plane(n: &Neighbors16x16) -> [[u8; 16]; 16] {
    // Plane mode — linear gradient fit, spec Section 8.3.3.4.
    // Requires all of top, left, and top-left to be available.
    let top = &n.top;
    let left = &n.left;
    let tl = n.top_left as i32;

    // H = sum_{i=0..=7} (i+1) * (top[8+i] - top[6-i])
    // but top[-1] in the spec is the top-left corner (our `tl`).
    // The spec uses P'[x,-1] for x in -1..=15, i.e. P'[-1,-1]=top_left and
    // P'[x,-1]=top[x] for x in 0..=15. So top[6-i] for i=7 refers to
    // top[-1] = top_left.
    let mut h = 0i32;
    for i in 0..8 {
        let t_hi = top[8 + i] as i32;
        let t_lo = if 6 >= i { top[6 - i] as i32 } else { tl }; // i=7 -> use tl
        h += (i as i32 + 1) * (t_hi - t_lo);
    }

    // V = sum_{j=0..=7} (j+1) * (left[8+j] - left[6-j])
    let mut v = 0i32;
    for j in 0..8 {
        let l_hi = left[8 + j] as i32;
        let l_lo = if 6 >= j { left[6 - j] as i32 } else { tl };
        v += (j as i32 + 1) * (l_hi - l_lo);
    }

    let b = (5 * h + 32) >> 6;
    let c = (5 * v + 32) >> 6;
    let a = 16 * (top[15] as i32 + left[15] as i32);

    let mut p = [[0u8; 16]; 16];
    for y in 0..16 {
        for x in 0..16 {
            let val = (a + b * (x as i32 - 7) + c * (y as i32 - 7) + 16) >> 5;
            p[y][x] = clip1(val);
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_4x4_neighbors() -> Neighbors4x4 {
        Neighbors4x4 {
            top: [100, 110, 120, 130, 140, 150, 160, 170],
            left: [80, 90, 100, 110],
            top_left: 90,
            top_available: true,
            top_right_available: true,
            left_available: true,
            top_left_available: true,
        }
    }

    #[test]
    fn pred4x4_vertical_mirrors_top() {
        let n = default_4x4_neighbors();
        let p = predict_4x4(Intra4x4Mode::Vertical, &n);
        for y in 0..4 {
            assert_eq!(p[y], [100, 110, 120, 130]);
        }
    }

    #[test]
    fn pred4x4_horizontal_mirrors_left() {
        let n = default_4x4_neighbors();
        let p = predict_4x4(Intra4x4Mode::Horizontal, &n);
        for y in 0..4 {
            assert_eq!(p[y], [n.left[y]; 4]);
        }
    }

    #[test]
    fn pred4x4_dc_equals_average_when_both_neighbors_available() {
        let n = default_4x4_neighbors();
        // sum(top[0..4]) + sum(left) = (100+110+120+130) + (80+90+100+110) = 840.
        // DC = (840 + 4) >> 3 = 844 / 8 = 105.
        let p = predict_4x4(Intra4x4Mode::Dc, &n);
        for row in &p {
            for &v in row {
                assert_eq!(v, 105);
            }
        }
    }

    #[test]
    fn pred4x4_dc_falls_back_to_128_when_no_neighbors() {
        let mut n = default_4x4_neighbors();
        n.top_available = false;
        n.left_available = false;
        let p = predict_4x4(Intra4x4Mode::Dc, &n);
        for row in &p {
            for &v in row {
                assert_eq!(v, 128);
            }
        }
    }

    #[test]
    fn pred4x4_dc_uses_only_top_when_left_missing() {
        let mut n = default_4x4_neighbors();
        n.left_available = false;
        // Average of top[0..4] = 115. (100+110+120+130)/4 = 115.
        // Spec rounds: (460 + 2) >> 2 = 115.
        let p = predict_4x4(Intra4x4Mode::Dc, &n);
        for row in &p {
            for &v in row {
                assert_eq!(v, 115);
            }
        }
    }

    #[test]
    fn pred16x16_vertical_mirrors_top() {
        let mut top = [0u8; 16];
        for i in 0..16 {
            top[i] = (i * 7) as u8;
        }
        let n = Neighbors16x16 {
            top,
            left: [50; 16],
            top_left: 0,
            top_available: true,
            left_available: true,
            top_left_available: true,
        };
        let p = predict_16x16(Intra16x16Mode::Vertical, &n);
        for y in 0..16 {
            assert_eq!(p[y], top);
        }
    }

    #[test]
    fn pred16x16_dc_averages_top_and_left() {
        // top = all 100, left = all 200. Total = 16*100 + 16*200 = 4800.
        // DC = (4800 + 16) >> 5 = 4816/32 = 150.
        let n = Neighbors16x16 {
            top: [100; 16],
            left: [200; 16],
            top_left: 0,
            top_available: true,
            left_available: true,
            top_left_available: true,
        };
        let p = predict_16x16(Intra16x16Mode::Dc, &n);
        for row in &p {
            for &v in row {
                assert_eq!(v, 150);
            }
        }
    }

    #[test]
    fn pred16x16_plane_on_flat_input_is_flat() {
        // All-flat neighbors should give a flat plane output.
        let n = Neighbors16x16 {
            top: [128; 16],
            left: [128; 16],
            top_left: 128,
            top_available: true,
            left_available: true,
            top_left_available: true,
        };
        let p = predict_16x16(Intra16x16Mode::Plane, &n);
        for row in &p {
            for &v in row {
                assert_eq!(v, 128, "flat input should produce flat plane");
            }
        }
    }

    fn default_chroma_neighbors() -> NeighborsChroma8x8 {
        NeighborsChroma8x8 {
            top: [100, 110, 120, 130, 140, 150, 160, 170],
            left: [80, 90, 100, 110, 120, 130, 140, 150],
            top_left: 90,
            top_available: true,
            left_available: true,
            top_left_available: true,
        }
    }

    #[test]
    fn pred_chroma_vertical_mirrors_top() {
        let n = default_chroma_neighbors();
        let p = predict_chroma_8x8(IntraChroma8x8Mode::Vertical, &n);
        for y in 0..8 {
            assert_eq!(p[y], n.top);
        }
    }

    #[test]
    fn pred_chroma_horizontal_mirrors_left() {
        let n = default_chroma_neighbors();
        let p = predict_chroma_8x8(IntraChroma8x8Mode::Horizontal, &n);
        for y in 0..8 {
            assert_eq!(p[y], [n.left[y]; 8]);
        }
    }

    #[test]
    fn pred_chroma_dc_produces_four_quadrant_values() {
        let n = default_chroma_neighbors();
        let p = predict_chroma_8x8(IntraChroma8x8Mode::Dc, &n);
        // Top-left quadrant: (sum(top[0..4]) + sum(left[0..4]) + 4) >> 3
        // = (460 + 380 + 4) >> 3 = 844/8 = 105. Each pixel in [0..4]x[0..4].
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(p[y][x], 105, "TL quadrant at ({y},{x})");
            }
        }
        // Top-right: (sum(top[4..8]) + 2) >> 2 = (620 + 2) >> 2 = 155.
        for y in 0..4 {
            for x in 4..8 {
                assert_eq!(p[y][x], 155, "TR quadrant at ({y},{x})");
            }
        }
        // Bottom-left: (sum(left[4..8]) + 2) >> 2 = (540 + 2) >> 2 = 135.
        for y in 4..8 {
            for x in 0..4 {
                assert_eq!(p[y][x], 135, "BL quadrant at ({y},{x})");
            }
        }
        // Bottom-right: (sum(top[4..8]) + sum(left[4..8]) + 4) >> 3 = 145.
        for y in 4..8 {
            for x in 4..8 {
                assert_eq!(p[y][x], 145, "BR quadrant at ({y},{x})");
            }
        }
    }

    #[test]
    fn pred_chroma_dc_fallback_to_128_when_no_neighbors() {
        let mut n = default_chroma_neighbors();
        n.top_available = false;
        n.left_available = false;
        let p = predict_chroma_8x8(IntraChroma8x8Mode::Dc, &n);
        for row in &p {
            for &v in row {
                assert_eq!(v, 128);
            }
        }
    }

    #[test]
    fn pred_chroma_plane_on_flat_input_is_flat() {
        let n = NeighborsChroma8x8 {
            top: [128; 8],
            left: [128; 8],
            top_left: 128,
            top_available: true,
            left_available: true,
            top_left_available: true,
        };
        let p = predict_chroma_8x8(IntraChroma8x8Mode::Plane, &n);
        for row in &p {
            for &v in row {
                assert_eq!(v, 128);
            }
        }
    }

    #[test]
    fn pred4x4_modes_all_return_valid_samples() {
        // Smoke test: every mode on a generic neighbor set produces 0..=255
        // sample values and doesn't panic.
        let n = default_4x4_neighbors();
        for m in [
            Intra4x4Mode::Vertical,
            Intra4x4Mode::Horizontal,
            Intra4x4Mode::Dc,
            Intra4x4Mode::DiagonalDownLeft,
            Intra4x4Mode::DiagonalDownRight,
            Intra4x4Mode::VerticalRight,
            Intra4x4Mode::HorizontalDown,
            Intra4x4Mode::VerticalLeft,
            Intra4x4Mode::HorizontalUp,
        ] {
            let p = predict_4x4(m, &n);
            for row in &p {
                for &v in row {
                    // u8 clamp is implicit — just verify it ran.
                    let _ = v;
                }
            }
        }
    }
}
