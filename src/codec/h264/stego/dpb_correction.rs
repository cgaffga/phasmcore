// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// P3.3a — post-frame DPB correction for inter-frame cascade elimination.
//
// After each frame's CABAC emit (Pass 2), the wire_only path has
// already written the stego bits to the bitstream but the encoder's
// pDecPic (reconstruction buffer) still holds pre-flip pixel data.
// The next P/B-frame's motion estimation reads pDecPic, so any
// coefficient flip that changed the pixel residual without updating
// pDecPic cascades: ME references stale data → predicted pixels
// differ from what the decoder will see → inter-frame quality cliff.
//
// This module computes the pixel-level delta that each flipped
// coefficient introduces and applies it to pDecPic's Y plane,
// closing the loop so ME reads post-flip reconstruction.
//
// Scope: CSB (CoeffSignBypass) + CSL (CoeffSuffixLsb) luma 4×4
// blocks only. MVD domains don't have a pixel-block
// interpretation; chroma correction is a follow-on (P3.3c).

use std::collections::HashMap;

use crate::codec::h264::macroblock::BLOCK_INDEX_TO_POS;
use crate::codec::h264::stego::hook::{BinKind, EmbedDomain, SyntaxPath};
use crate::codec::h264::stego::inject::DomainCover;
use crate::codec::h264::tables::ZIGZAG_4X4;

// ─── H.264 dequant scale (spec § 8.5.8, Eq. 8-317/319) ──────────
//
// LevelScale4x4[qp%6][class] — same table as transform.rs but
// duplicated here to avoid making the crate-private table public
// for a single consumer. If the canonical table ever changes, this
// must track.
const LEVEL_SCALE_4X4: [[i32; 3]; 6] = [
    [160, 256, 208],
    [176, 288, 224],
    [208, 320, 256],
    [224, 368, 288],
    [256, 400, 320],
    [288, 464, 368],
];

/// Classify a 4×4 coefficient position into one of the three
/// LevelScale classes (same as `transform::norm_adjust_class`).
#[inline]
const fn norm_adjust_class(i: usize, j: usize) -> usize {
    let even_i = i & 1 == 0;
    let even_j = j & 1 == 0;
    if even_i && even_j {
        0
    } else if !even_i && !even_j {
        1
    } else {
        2
    }
}

// ─── Pixel-delta computation ─────────────────────────────────────

/// Compute the 4×4 pixel delta block that results from changing a
/// single quantized coefficient at zigzag scan position `scan_pos`
/// by `delta_coeff` at the given QP.
///
/// Uses the H.264 4×4 integer inverse transform (spec § 8.5.12.1)
/// applied to a block with a single non-zero coefficient at the
/// affected position. The result is the signed pixel-level change.
///
/// Returns the 4×4 delta block in row-major order.
fn compute_single_coeff_pixel_delta(
    scan_pos: u8,
    delta_coeff: i32,
    qp: i32,
) -> [[i32; 4]; 4] {
    // Map zigzag scan position to raster (row, col) in the 4×4 block.
    let raster_idx = ZIGZAG_4X4[scan_pos as usize] as usize;
    let u = raster_idx / 4; // row
    let v = raster_idx % 4; // col

    // Build a 4×4 block with a single non-zero coefficient at (u,v),
    // already dequantized per spec § 8.5.12.2.
    let q_mod = qp.rem_euclid(6) as usize;
    let q_bits = qp / 6;
    let scale = LEVEL_SCALE_4X4[q_mod][norm_adjust_class(u, v)];

    let mut d = [[0i32; 4]; 4];
    if q_bits >= 4 {
        d[u][v] = (delta_coeff * scale) << (q_bits - 4);
    } else {
        let shift = 4 - q_bits;
        let rnd = 1 << (shift - 1);
        d[u][v] = (delta_coeff * scale + rnd) >> shift;
    }

    // Run the standard 4×4 integer inverse transform (same butterfly
    // as `transform::inverse_4x4_integer`).
    //
    // Stage 1: 1-D inverse along columns.
    let mut g = [[0i32; 4]; 4];
    for j in 0..4 {
        let e0 = d[0][j] + d[2][j];
        let e1 = d[0][j] - d[2][j];
        let e2 = (d[1][j] >> 1) - d[3][j];
        let e3 = d[1][j] + (d[3][j] >> 1);
        g[0][j] = e0 + e3;
        g[1][j] = e1 + e2;
        g[2][j] = e1 - e2;
        g[3][j] = e0 - e3;
    }

    // Stage 2: 1-D inverse along rows.
    let mut h = [[0i32; 4]; 4];
    for i in 0..4 {
        let e0 = g[i][0] + g[i][2];
        let e1 = g[i][0] - g[i][2];
        let e2 = (g[i][1] >> 1) - g[i][3];
        let e3 = g[i][1] + (g[i][3] >> 1);
        h[i][0] = e0 + e3;
        h[i][1] = e1 + e2;
        h[i][2] = e1 - e2;
        h[i][3] = e0 - e3;
    }

    // Stage 3: rounding right-shift by 6.
    let mut r = [[0i32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            r[i][j] = (h[i][j] + 32) >> 6;
        }
    }
    r
}

// ─── IDCT delta cache ───────────────────────────────────────────

/// Pre-computed IDCT pixel-delta blocks keyed by (scan_pos, delta_coeff).
/// QP is fixed at construction. Avoids recomputing the IDCT butterfly
/// (~400 ALU ops) for repeated (scan_pos, delta_coeff) pairs across
/// a frame or video. Typical size: ~500 entries × 64 B = ~32 KB.
pub struct IdctDeltaCache {
    qp: i32,
    map: HashMap<(u8, i32), [[i32; 4]; 4]>,
}

impl IdctDeltaCache {
    pub fn new(qp: i32) -> Self {
        Self { qp, map: HashMap::with_capacity(256) }
    }

    #[inline]
    pub fn get(&mut self, scan_pos: u8, delta_coeff: i32) -> &[[i32; 4]; 4] {
        self.map.entry((scan_pos, delta_coeff))
            .or_insert_with(|| compute_single_coeff_pixel_delta(scan_pos, delta_coeff, self.qp))
    }
}

// ─── Public API ──────────────────────────────────────────────────

/// Compute the pixel-level deltas introduced by all CSB + CSL
/// overrides for a given frame and apply them to the encoder's
/// pDecPic Y plane.
///
/// # Arguments
///
/// * `dec_pic_y`     — mutable slice over pDecPic's Y plane (stride-
///                     padded rows, `stride * height` bytes).
/// * `dec_pic_stride` — byte stride of pDecPic's Y plane (may be
///                      wider than `width` due to encoder padding).
/// * `width`, `height` — frame dimensions in pixels (must be
///                        multiples of 16).
/// * `override_map`  — the STC plan's override entries keyed by
///                      `PositionKey::raw()`. Contains ONLY entries
///                      where the stego bit differs from the cover
///                      bit (i.e., positions that were actually
///                      flipped).
/// * `cover`         — the 4-domain cover from the Pass 1 walker.
///                      Provides the cover bits + position keys for
///                      every candidate CSB/CSL position.
/// * `frame_idx`     — which frame to process (filters positions by
///                      `PositionKey::frame_idx()`).
/// * `qp`            — the effective luma QP for this frame.
///
/// # What it does
///
/// For each CSB/CSL position in the cover that belongs to
/// `frame_idx` AND has an override entry in `override_map`:
///
/// 1. Determine the coefficient delta:
///    - CSB (sign flip): `delta_coeff = -2 * magnitude` (coefficient
///      swings from +M to -M or vice versa).
///    - CSL (suffix LSB flip): `delta_coeff = ±1` (magnitude changes
///      by 1).
///
/// 2. Compute the 4×4 pixel delta via inverse-quantize + IDCT on a
///    single-coefficient block.
///
/// 3. Add the pixel delta to the corresponding 4×4 region of
///    `dec_pic_y`, clamping to [0, 255].
pub fn compute_and_apply_deltas(
    dec_pic_y: &mut [u8],
    dec_pic_stride: usize,
    width: u32,
    height: u32,
    override_map: &HashMap<u64, u8>,
    cover: &DomainCover,
    frame_idx: u32,
    qp: i32,
) {
    if override_map.is_empty() {
        return;
    }

    let mb_width = (width / 16) as usize;
    let pic_h = height as usize;
    let mut cache = IdctDeltaCache::new(qp);

    process_domain_csb(
        dec_pic_y, dec_pic_stride, mb_width, pic_h,
        override_map, &cover.coeff_sign_bypass.positions,
        &cover.coeff_sign_bypass.bits, &cover.coeff_sign_bypass.magnitudes,
        frame_idx, &mut cache,
    );

    process_domain_csl(
        dec_pic_y, dec_pic_stride, mb_width, pic_h,
        override_map, &cover.coeff_suffix_lsb.positions,
        &cover.coeff_suffix_lsb.bits, &cover.coeff_suffix_lsb.magnitudes,
        frame_idx, &mut cache,
    );
}

/// Process all CSB positions for the given frame.
fn process_domain_csb(
    dec_pic_y: &mut [u8],
    stride: usize,
    mb_width: usize,
    pic_h: usize,
    override_map: &HashMap<u64, u8>,
    positions: &[super::hook::PositionKey],
    cover_bits: &[u8],
    magnitudes: &[u16],
    frame_idx: u32,
    cache: &mut IdctDeltaCache,
) {
    for (i, &pos) in positions.iter().enumerate() {
        if pos.frame_idx() != frame_idx {
            continue;
        }
        let raw = pos.raw();
        let Some(&stego_bit) = override_map.get(&raw) else { continue };
        if stego_bit == cover_bits[i] {
            continue;
        }
        if pos.domain() != EmbedDomain::CoeffSignBypass {
            continue;
        }
        let (block_idx, coeff_idx) = match pos.syntax_path() {
            SyntaxPath::Luma4x4 { block_idx, coeff_idx, kind: BinKind::Sign } => {
                (block_idx, coeff_idx)
            }
            _ => continue,
        };
        let magnitude = magnitudes[i] as i32;
        if magnitude == 0 {
            continue;
        }
        let delta_coeff = -2 * magnitude;
        let pixel_delta = *cache.get(coeff_idx, delta_coeff);
        apply_delta_to_plane(
            dec_pic_y, stride, mb_width, pic_h,
            pos.mb_addr(), block_idx,
            &pixel_delta,
        );
    }
}

/// Process all CSL positions for the given frame.
fn process_domain_csl(
    dec_pic_y: &mut [u8],
    stride: usize,
    mb_width: usize,
    pic_h: usize,
    override_map: &HashMap<u64, u8>,
    positions: &[super::hook::PositionKey],
    cover_bits: &[u8],
    magnitudes: &[u16],
    frame_idx: u32,
    cache: &mut IdctDeltaCache,
) {
    for (i, &pos) in positions.iter().enumerate() {
        if pos.frame_idx() != frame_idx {
            continue;
        }
        let raw = pos.raw();
        let Some(&stego_bit) = override_map.get(&raw) else { continue };
        if stego_bit == cover_bits[i] {
            continue;
        }
        if pos.domain() != EmbedDomain::CoeffSuffixLsb {
            continue;
        }
        let (block_idx, coeff_idx) = match pos.syntax_path() {
            SyntaxPath::Luma4x4 { block_idx, coeff_idx, kind: BinKind::SuffixLsb } => {
                (block_idx, coeff_idx)
            }
            _ => continue,
        };

        // CSL suffix-LSB flip: magnitude changes by ±1.
        // Direction: at threshold (16), only +1 is valid; otherwise
        // prefer -1 (toward zero, lower distortion). This matches
        // inject.rs::flipped_magnitude.
        let abs_mag = magnitudes[i] as i32;
        if abs_mag == 0 {
            continue;
        }
        // The delta_coeff for CSL is ±1. The sign of the coefficient
        // didn't change, only its magnitude. We need the direction:
        // if |coeff| was at the threshold (16), the flip went +1;
        // otherwise -1. The cover_bit encodes the old LSB state
        // (suffix_lsb_bit_for_magnitude), and the stego_bit is the
        // new one. The magnitude change is:
        //   old_abs → new_abs where new_abs = flipped_magnitude(old_abs)
        //   delta_coeff = sign(coeff) * (new_abs - old_abs)
        //
        // But we only have |coeff|. Since sign is preserved, the
        // pixel delta from changing |coeff| by d is the same as from
        // changing the signed coefficient by sign * d. However, for
        // the IDCT the sign of delta_coeff matters:
        //   if coeff > 0: delta_coeff = +(new_abs - old_abs)
        //   if coeff < 0: delta_coeff = -(new_abs - old_abs)
        //
        // We don't know the sign of the coefficient from the cover
        // data. But the magnitude delta itself is small (±1) and the
        // sign determines which direction in pixel space. We can
        // determine sign from the cover_bit:
        //   cover_bit for CSB at the same position would be 0 if
        //   positive, 1 if negative.
        //
        // However, we don't have access to the CSB bit for this
        // specific coefficient. The conservative approach: compute
        // the pixel delta for delta_coeff = +1 (magnitude increase)
        // or delta_coeff = -1 (magnitude decrease). We know the
        // direction from the threshold rule.
        //
        // Actually, the direction is deterministic from abs_mag:
        //   abs_mag == 16 (threshold) → new_abs = 17, delta = +1
        //   abs_mag != 16            → new_abs = abs_mag - 1, delta = -1
        //
        // The IDCT is linear, so the pixel delta for a coefficient
        // changing from C to C+d equals the IDCT of a block with
        // just d at that position, regardless of the coefficient's
        // sign. The sign of the coefficient only determines whether
        // the dequantized value is positive or negative, but the
        // CHANGE in dequantized value depends only on delta.
        let delta_coeff: i32 = if abs_mag == 16 { 1 } else { -1 };

        let pixel_delta = *cache.get(coeff_idx, delta_coeff);
        apply_delta_to_plane(
            dec_pic_y, stride, mb_width, pic_h,
            pos.mb_addr(), block_idx,
            &pixel_delta,
        );
    }
}

/// Apply a 4×4 pixel delta block to the Y plane at the position
/// specified by `mb_addr` + `block_idx`.
#[inline]
fn apply_delta_to_plane(
    dec_pic_y: &mut [u8],
    stride: usize,
    mb_width: usize,
    pic_h: usize,
    mb_addr: u32,
    block_idx: u8,
    pixel_delta: &[[i32; 4]; 4],
) {
    let mb_x = (mb_addr as usize) % mb_width;
    let mb_y = (mb_addr as usize) / mb_width;

    let (bx, by) = BLOCK_INDEX_TO_POS[block_idx as usize];
    let px_x = mb_x * 16 + bx as usize * 4;
    let px_y = mb_y * 16 + by as usize * 4;

    for di in 0..4 {
        let y = px_y + di;
        if y >= pic_h {
            break;
        }
        for dj in 0..4 {
            let x = px_x + dj;
            // Width check: px_x is at most (mb_width-1)*16 + 3*4 + 3,
            // which for valid MB dimensions is always within the stride.
            // But guard defensively.
            let off = y * stride + x;
            if off >= dec_pic_y.len() {
                continue;
            }
            let old = dec_pic_y[off] as i32;
            let new_val = (old + pixel_delta[di][dj]).clamp(0, 255);
            dec_pic_y[off] = new_val as u8;
        }
    }
}

/// D2.2 — per-row variant: apply IDCT deltas only for MBs whose
/// `mb_addr / mb_width == mb_row`. Called from the row-complete
/// callback so the NEXT row's vertical intra prediction reads
/// post-flip pixels within the same frame.
pub fn compute_and_apply_deltas_for_row(
    dec_pic_y: &mut [u8],
    dec_pic_stride: usize,
    width: u32,
    height: u32,
    override_map: &HashMap<u64, u8>,
    cover: &DomainCover,
    frame_idx: u32,
    mb_row: u32,
    qp: i32,
) {
    if override_map.is_empty() {
        return;
    }

    let mb_width = (width / 16) as usize;
    let pic_h = height as usize;
    let mut cache = IdctDeltaCache::new(qp);

    process_domain_csb_row(
        dec_pic_y, dec_pic_stride, mb_width, pic_h,
        override_map, &cover.coeff_sign_bypass.positions,
        &cover.coeff_sign_bypass.bits, &cover.coeff_sign_bypass.magnitudes,
        frame_idx, mb_row, &mut cache,
    );

    process_domain_csl_row(
        dec_pic_y, dec_pic_stride, mb_width, pic_h,
        override_map, &cover.coeff_suffix_lsb.positions,
        &cover.coeff_suffix_lsb.bits, &cover.coeff_suffix_lsb.magnitudes,
        frame_idx, mb_row, &mut cache,
    );
}

fn process_domain_csb_row(
    dec_pic_y: &mut [u8],
    stride: usize,
    mb_width: usize,
    pic_h: usize,
    override_map: &HashMap<u64, u8>,
    positions: &[super::hook::PositionKey],
    cover_bits: &[u8],
    magnitudes: &[u16],
    frame_idx: u32,
    mb_row: u32,
    cache: &mut IdctDeltaCache,
) {
    for (i, &pos) in positions.iter().enumerate() {
        if pos.frame_idx() != frame_idx {
            continue;
        }
        if (pos.mb_addr() as usize) / mb_width != mb_row as usize {
            continue;
        }
        let raw = pos.raw();
        let Some(&stego_bit) = override_map.get(&raw) else { continue };
        if stego_bit == cover_bits[i] {
            continue;
        }
        if pos.domain() != EmbedDomain::CoeffSignBypass {
            continue;
        }
        let (block_idx, coeff_idx) = match pos.syntax_path() {
            SyntaxPath::Luma4x4 { block_idx, coeff_idx, kind: BinKind::Sign } => {
                (block_idx, coeff_idx)
            }
            _ => continue,
        };
        let magnitude = magnitudes[i] as i32;
        if magnitude == 0 {
            continue;
        }
        let delta_coeff = -2 * magnitude;
        let pixel_delta = *cache.get(coeff_idx, delta_coeff);
        apply_delta_to_plane(
            dec_pic_y, stride, mb_width, pic_h,
            pos.mb_addr(), block_idx, &pixel_delta,
        );
    }
}

fn process_domain_csl_row(
    dec_pic_y: &mut [u8],
    stride: usize,
    mb_width: usize,
    pic_h: usize,
    override_map: &HashMap<u64, u8>,
    positions: &[super::hook::PositionKey],
    cover_bits: &[u8],
    magnitudes: &[u16],
    frame_idx: u32,
    mb_row: u32,
    cache: &mut IdctDeltaCache,
) {
    for (i, &pos) in positions.iter().enumerate() {
        if pos.frame_idx() != frame_idx {
            continue;
        }
        if (pos.mb_addr() as usize) / mb_width != mb_row as usize {
            continue;
        }
        let raw = pos.raw();
        let Some(&stego_bit) = override_map.get(&raw) else { continue };
        if stego_bit == cover_bits[i] {
            continue;
        }
        if pos.domain() != EmbedDomain::CoeffSuffixLsb {
            continue;
        }
        let (block_idx, coeff_idx) = match pos.syntax_path() {
            SyntaxPath::Luma4x4 { block_idx, coeff_idx, kind: BinKind::SuffixLsb } => {
                (block_idx, coeff_idx)
            }
            _ => continue,
        };
        let abs_mag = magnitudes[i] as i32;
        if abs_mag == 0 {
            continue;
        }
        let delta_coeff: i32 = if abs_mag == 16 { 1 } else { -1 };
        let pixel_delta = *cache.get(coeff_idx, delta_coeff);
        apply_delta_to_plane(
            dec_pic_y, stride, mb_width, pic_h,
            pos.mb_addr(), block_idx, &pixel_delta,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::transform::{dequant_4x4, inverse_4x4_integer, unzigzag_4x4};

    /// Verify that our single-coefficient IDCT delta matches the full
    /// `transform.rs` path for each of the 16 scan positions.
    #[test]
    fn single_coeff_delta_matches_full_transform() {
        for qp in [20, 26, 30, 34, 40] {
            for scan_pos in 0..16u8 {
                let delta_coeff = 3i32;
                // Our function.
                let our = compute_single_coeff_pixel_delta(scan_pos, delta_coeff, qp);

                // Reference: build a 16-element scan array with a single
                // non-zero coefficient, then run unzigzag + dequant + IDCT.
                let mut scan = [0i32; 16];
                scan[scan_pos as usize] = delta_coeff;
                let raster = unzigzag_4x4(&scan);
                let dequant = dequant_4x4(&raster, qp, false);
                let reference = inverse_4x4_integer(&dequant);

                for i in 0..4 {
                    for j in 0..4 {
                        assert_eq!(
                            our[i][j], reference[i][j],
                            "mismatch at scan_pos={scan_pos} qp={qp} pixel=({i},{j}): \
                             ours={} ref={}",
                            our[i][j], reference[i][j],
                        );
                    }
                }
            }
        }
    }

    /// Zero delta_coeff produces an all-zero pixel delta.
    #[test]
    fn zero_delta_produces_zero_pixels() {
        for scan_pos in 0..16u8 {
            let d = compute_single_coeff_pixel_delta(scan_pos, 0, 26);
            for row in &d {
                for &v in row {
                    assert_eq!(v, 0);
                }
            }
        }
    }

    /// Linearity: IDCT(2*x) == 2 * IDCT(x) up to rounding.
    #[test]
    fn linearity_within_rounding() {
        let qp = 26;
        let scan_pos = 5u8; // arbitrary AC position
        let d1 = compute_single_coeff_pixel_delta(scan_pos, 1, qp);
        let d2 = compute_single_coeff_pixel_delta(scan_pos, 2, qp);
        for i in 0..4 {
            for j in 0..4 {
                let diff = (d2[i][j] - 2 * d1[i][j]).abs();
                assert!(
                    diff <= 1,
                    "linearity violated at ({i},{j}): d2={} 2*d1={} diff={diff}",
                    d2[i][j], 2 * d1[i][j],
                );
            }
        }
    }
}
