// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Flood detection for decode-and-replan (spec
//! `docs/design/video/stego-cover-order-and-damage-costs.md` § 7).
//!
//! The encoder predicts from its natural reconstruction while a decoder sees
//! the flipped one, so an unlucky flip can leave a near-constant offset that
//! intra and inter prediction carry over a whole region for the rest of the
//! GOP (a *flood*). After a GOP is embedded, its stego bytes are decoded and
//! each frame's 16×16 luma block sums are compared with the natural
//! reconstruction's. A flooded frame names the region whose cover positions
//! become wet before the GOP is embedded again.
//!
//! Integer arithmetic only: the replan changes which positions flip, so it
//! must decide identically on every platform.

/// Block size of the luma grid, in pixels.
pub const FLOOD_BLOCK: usize = 16;

/// Per-frame luma block sums: `sums[by * bw + bx]` is the sum of the
/// (possibly partial, at the right and bottom edge) `FLOOD_BLOCK`-square
/// block at block column `bx`, row `by`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LumaBlockGrid {
    pub width: usize,
    pub height: usize,
    pub bw: usize,
    pub bh: usize,
    pub sums: Vec<u32>,
}

impl LumaBlockGrid {
    /// Block sums of a `width`×`height` luma plane with row stride `stride`.
    pub fn from_plane(y: &[u8], width: usize, height: usize, stride: usize) -> Self {
        let bw = width.div_ceil(FLOOD_BLOCK);
        let bh = height.div_ceil(FLOOD_BLOCK);
        let mut sums = vec![0u32; bw * bh];
        for row in 0..height {
            let line = &y[row * stride..row * stride + width];
            let base = (row / FLOOD_BLOCK) * bw;
            for (bx, chunk) in line.chunks(FLOOD_BLOCK).enumerate() {
                sums[base + bx] += chunk.iter().map(|&v| v as u32).sum::<u32>();
            }
        }
        Self { width, height, bw, bh, sums }
    }

    /// Pixels in block (bx, by): smaller at the right and bottom edge.
    fn block_pixels(&self, bx: usize, by: usize) -> i64 {
        let w = (self.width - bx * FLOOD_BLOCK).min(FLOOD_BLOCK);
        let h = (self.height - by * FLOOD_BLOCK).min(FLOOD_BLOCK);
        (w * h) as i64
    }
}

/// A local flood: a block whose mean shifts by at least this many levels.
/// Flip noise moves 16×16 block means by a few levels, rarely past 16
/// (IMG_4138 at the 720p red end: 99.9 % of blocks under 14); a prediction
/// mismatch carried over a region shifts it by 24–36, a dark or light
/// blotch that follows the motion for several frames. 20 levels puts the
/// block near 22 dB, under the 25 dB the quality gate counts as damage.
pub const LOCAL_FLOOD_LEVELS: i64 = 20;

/// A flooded frame, the luma-pixel rectangle `[x0, x1) × [y0, y1)` of its
/// shifted blocks, and `(sx, sy)`, the top-left pixel of the first shifted
/// block in raster order: where the flood entered, since intra prediction
/// carries an offset right, down and down-left (from the above-right edge),
/// never up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloodRegion {
    pub frame: usize,
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    pub sx: usize,
    pub sy: usize,
}

/// The first frame (decode order) whose stego luma carries a flood, or None.
///
/// From the block-mean shifts d_b = (stego − natural) / pixels, a frame is
/// flooded when either
/// - **whole frame:** |mean d| ≥ 0.5 levels and |mean d| ≥ 0.8 · mean |d_b|,
///   a shift of one sign over most of the frame (the quality gate flags
///   |mean| ≥ 1; replan reacts at half that); the region is the blocks
///   shifted by ≥ 1 level in the dominant direction;
/// - **local:** some block shifts by ≥ [`LOCAL_FLOOD_LEVELS`]; the region is
///   those blocks.
///
/// Zero-mean flip noise largely cancels within a block, so it trips
/// neither.
pub fn flood_region(natural: &[LumaBlockGrid], stego: &[LumaBlockGrid]) -> Option<FloodRegion> {
    natural.iter().zip(stego).enumerate().find_map(|(frame, (n, s))| frame_flood(frame, n, s))
}

/// How badly a decode floods, for choosing among replan rounds:
/// `(flooded frames, largest |block-mean shift| in 1/256 levels)`, compared
/// lexicographically (fewer flooded frames first, then the smaller worst
/// block).
pub fn flood_severity(natural: &[LumaBlockGrid], stego: &[LumaBlockGrid]) -> (usize, i64) {
    let mut flooded = 0;
    let mut worst = 0i64;
    for (frame, (n, s)) in natural.iter().zip(stego).enumerate() {
        if n.bw != s.bw || n.bh != s.bh || n.sums.len() != s.sums.len() {
            continue;
        }
        flooded += frame_flood(frame, n, s).is_some() as usize;
        for by in 0..n.bh {
            for bx in 0..n.bw {
                let i = by * n.bw + bx;
                let d = (s.sums[i] as i64 - n.sums[i] as i64).abs();
                worst = worst.max(d * 256 / n.block_pixels(bx, by).max(1));
            }
        }
    }
    (flooded, worst)
}

/// [`flood_region`]'s test on one frame.
fn frame_flood(frame: usize, n: &LumaBlockGrid, s: &LumaBlockGrid) -> Option<FloodRegion> {
    if n.bw != s.bw || n.bh != s.bh || n.sums.len() != s.sums.len() {
        return None;
    }
    let total_px = (n.width * n.height) as i64;
    let (mut shift, mut abs_shift) = (0i64, 0i64);
    for (&a, &b) in n.sums.iter().zip(&s.sums) {
        let d = b as i64 - a as i64;
        shift += d;
        abs_shift += d.abs();
    }
    // |mean| >= 0.5 and |mean| >= 0.8 * mean |block shift|, in integers.
    let whole = total_px > 0 && 2 * shift.abs() >= total_px && 10 * shift.abs() >= 8 * abs_shift;
    let sign = shift.signum();
    let shifted = |bx: usize, by: usize| -> bool {
        let i = by * n.bw + bx;
        let d = s.sums[i] as i64 - n.sums[i] as i64;
        let px = n.block_pixels(bx, by);
        if whole {
            d.signum() == sign && d.abs() >= px
        } else {
            d.abs() >= LOCAL_FLOOD_LEVELS * px
        }
    };
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    let mut source: Option<(usize, usize)> = None;
    for by in 0..n.bh {
        for bx in 0..n.bw {
            if !shifted(bx, by) {
                continue;
            }
            source.get_or_insert((bx * FLOOD_BLOCK, by * FLOOD_BLOCK));
            x0 = x0.min(bx * FLOOD_BLOCK);
            y0 = y0.min(by * FLOOD_BLOCK);
            x1 = x1.max(((bx + 1) * FLOOD_BLOCK).min(n.width));
            y1 = y1.max(((by + 1) * FLOOD_BLOCK).min(n.height));
        }
    }
    match source {
        Some((sx, sy)) => Some(FloodRegion { frame, x0, y0, x1, y1, sx, sy }),
        // Spread too thin for any block to shift by a whole level: the whole
        // frame is the region.
        None if whole => Some(FloodRegion { frame, x0: 0, y0: 0, x1: n.width, y1: n.height, sx: 0, sy: 0 }),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> Vec<u8> {
        (0..w * h).map(|i| f(i % w, i / w)).collect()
    }

    #[test]
    fn block_sums_cover_partial_edge_blocks() {
        let (w, h) = (40, 20);
        let g = LumaBlockGrid::from_plane(&plane(w, h, |_, _| 2), w, h, w);
        assert_eq!((g.bw, g.bh), (3, 2));
        assert_eq!(g.sums[0], 2 * 16 * 16);
        assert_eq!(g.sums[2], 2 * 8 * 16); // right edge: 8 px wide
        assert_eq!(g.sums[5], 2 * 8 * 4); // corner: 8 × 4
    }

    #[test]
    fn block_sums_honour_the_stride() {
        let (w, h, stride) = (16, 16, 24);
        let mut y = vec![9u8; stride * h];
        for r in 0..h {
            for c in 0..w {
                y[r * stride + c] = 1;
            }
        }
        assert_eq!(LumaBlockGrid::from_plane(&y, w, h, stride).sums, vec![256]);
    }

    fn grids(offset: impl Fn(usize, usize) -> i32, frames: usize) -> (Vec<LumaBlockGrid>, Vec<LumaBlockGrid>) {
        let (w, h) = (128, 96);
        let nat = plane(w, h, |x, y| ((x * 3 + y * 5) % 120 + 60) as u8);
        let st: Vec<u8> = nat
            .iter()
            .enumerate()
            .map(|(i, &v)| (v as i32 + offset(i % w, i / w)).clamp(0, 255) as u8)
            .collect();
        let n = LumaBlockGrid::from_plane(&nat, w, h, w);
        let s = LumaBlockGrid::from_plane(&st, w, h, w);
        let mut ns = vec![n.clone(); frames];
        let mut ss = vec![n; frames];
        ns[frames - 1] = LumaBlockGrid::from_plane(&nat, w, h, w);
        ss[frames - 1] = s;
        (ns, ss)
    }

    #[test]
    fn flood_region_detects_a_wedge_offset() {
        // +6 right of and below (64, 32), as intra prediction spreads it.
        let (n, s) = grids(|x, y| if x >= 64 && y >= 32 { 6 } else { 0 }, 3);
        let r = flood_region(&n, &s).expect("flood");
        assert_eq!(r.frame, 2);
        assert_eq!((r.x0, r.y0), (64, 32));
        assert_eq!((r.x1, r.y1), (128, 96));
    }

    #[test]
    fn flood_region_ignores_zero_mean_noise() {
        // ±8 in a checkerboard of pixels: large errors, no block shift.
        let (n, s) = grids(|x, y| if (x + y) % 2 == 0 { 8 } else { -8 }, 2);
        assert_eq!(flood_region(&n, &s), None);
    }

    #[test]
    fn flood_region_ignores_a_small_local_shift() {
        // One 16×16 block shifted by 10: within what flip noise reaches.
        let (n, s) = grids(|x, y| if x < 16 && y < 16 { 10 } else { 0 }, 2);
        assert_eq!(flood_region(&n, &s), None);
    }

    #[test]
    fn flood_region_detects_a_local_flood() {
        // A 32×16 patch 26 levels dark in an otherwise clean frame: the
        // IMG_4138 blotch (frames 444–449, capped red end), which the
        // whole-frame test missed.
        let (n, s) = grids(|x, y| if (48..80).contains(&x) && (32..48).contains(&y) { -26 } else { 0 }, 2);
        let r = flood_region(&n, &s).expect("local flood");
        assert_eq!(r.frame, 1);
        assert_eq!((r.x0, r.y0, r.x1, r.y1), (48, 32, 80, 48));
        assert_eq!((r.sx, r.sy), (48, 32));
    }

    #[test]
    fn flood_source_is_the_first_shifted_block_in_raster_order() {
        // An offset entering around (64, 16) and spreading down and to the
        // left (intra prediction reads the above-right edge): the bounding
        // box's corner lands at the frame's left edge, four rows below the
        // entry; the first shifted block in raster order is at the entry.
        let (n, s) = grids(|x, y| if y >= 16 && x + (y - 16) >= 64 { 6 } else { 0 }, 2);
        let r = flood_region(&n, &s).expect("flood");
        assert_eq!((r.x0, r.y0), (0, 16));
        assert_eq!((r.sx, r.sy), (48, 16));
    }

    #[test]
    fn severity_ranks_floods_then_the_largest_shift() {
        let clean = grids(|_, _| 0, 2);
        let noise = grids(|x, y| if x < 16 && y < 16 { 10 } else { 0 }, 2);
        let local = grids(|x, y| if x < 16 && y < 16 { -26 } else { 0 }, 2);
        let wedge = grids(|x, y| if x >= 64 && y >= 32 { 6 } else { 0 }, 2);
        assert_eq!(flood_severity(&clean.0, &clean.1), (0, 0));
        // No flood, but a 10-level block: (0, 10 levels in 1/256).
        assert_eq!(flood_severity(&noise.0, &noise.1), (0, 10 * 256));
        assert_eq!(flood_severity(&local.0, &local.1), (1, 26 * 256));
        assert_eq!(flood_severity(&wedge.0, &wedge.1).0, 1);
        assert!(flood_severity(&noise.0, &noise.1) < flood_severity(&local.0, &local.1));
    }

    #[test]
    fn identical_frames_have_no_flood() {
        let (n, _) = grids(|_, _| 0, 2);
        assert_eq!(flood_region(&n, &n), None);
    }
}
