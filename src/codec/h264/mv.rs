// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 motion vector parsing and storage for P-slice macroblocks.
//!
//! Phase 2 needs actual MV fields — not just consumed-and-discarded MVDs —
//! so later phases (DDCA inter-frame propagation, Phase 3 MVD embedding)
//! can reason about which regions of reference frames a P-block depends on.
//!
//! This module implements:
//! * `MotionVector` / `MvField` — per-4×4-block storage per MB (list-0 only;
//!   H.264 Baseline has no B-slices).
//! * `MvPredictorContext` — a frame-wide 4×4-granular grid tracking the
//!   resolved MV + ref_idx of every P-block already decoded.
//! * `median_mv` — the H.264 Section 8.4.1.3 median predictor with the
//!   single-matching-neighbour special case.
//! * `parse_mv_field` — reads MVDs + ref_idx from the bitstream, runs the
//!   predictor, and stores the absolute MVs into both the output `MvField`
//!   and the frame-wide `MvPredictorContext`.
//!
//! The neighbour lookup follows spec 8.4.1.3 closely but uses the
//! top-left 4×4 block of each partition as the lookup anchor — correct for
//! most partitions and a sound approximation for all Baseline partition
//! sizes. Phase 3's MVD embedding will need bit-accurate predictors; when
//! that work lands we can tighten this up.

use super::bitstream::{EpByteMap, RbspReader};
use super::cavlc::{check_ep_conflict, EmbedDomain, EmbeddablePosition};
use super::macroblock::MbType;
use super::H264Error;

/// Sentinel `block_idx` value used by Phase 3 MVD positions. Mirrors the
/// `u32::MAX` sentinel already used for I_16x16 DC WET positions: cost
/// functions branch on it, the pipeline leaves it unshifted when promoting
/// positions to frame-global indices.
pub const MVD_BLOCK_IDX_SENTINEL: u32 = u32::MAX - 1;

/// Capture the suffix-LSB position of a single signed-Exp-Golomb MVD reader
/// pair (`bits_before` / `bits_after` bracket one `read_se` call).
///
/// H.264 signed Exp-Golomb codeword: `0^lz 1 suffix[lz bits]`, total length
/// `2·lz + 1`. codeNum = 0 (mvd = 0) has `lz = 0` and **no suffix** — we
/// return `None`. For codeNum ≥ 1, the LSB of the suffix is the last bit of
/// the codeword (at RBSP bit offset `bits_before + 2·lz`).
///
/// Flipping that bit changes codeNum by ±1 while keeping the codeword length
/// unchanged → downstream parsing is unaffected, the MV value shifts by one
/// quarter-pel unit in the mapped domain.
fn capture_mvd_position(
    bits_before: usize,
    bits_after: usize,
    mvd_value: i16,
    ep_map: &EpByteMap,
    raw_data: &[u8],
) -> Option<EmbeddablePosition> {
    if mvd_value == 0 {
        // codeNum == 0 → codeword is just the single bit `1`, no suffix.
        return None;
    }
    let len = bits_after.saturating_sub(bits_before);
    if len < 3 || len.is_multiple_of(2) {
        // Sanity: codeword length must be 2·lz + 1 with lz ≥ 1.
        return None;
    }
    // Suffix LSB sits at the last bit of the codeword.
    let lsb_bit_idx = bits_after - 1;
    let rbsp_byte = lsb_bit_idx / 8;
    let rbsp_bit = (lsb_bit_idx % 8) as u8;
    if rbsp_byte >= ep_map.rbsp_to_raw.len() {
        return None;
    }
    let raw_byte = ep_map.rbsp_to_raw[rbsp_byte];
    let ep_conflict = check_ep_conflict(raw_data, raw_byte, rbsp_bit);
    Some(EmbeddablePosition {
        raw_byte_offset: raw_byte,
        bit_offset: rbsp_bit,
        domain: EmbedDomain::MvdLsb,
        scan_pos: 0,
        coeff_value: mvd_value as i32,
        ep_conflict,
        block_idx: MVD_BLOCK_IDX_SENTINEL,
        frame_idx: 0,
        mb_idx: 0, // Set by pipeline at position-shift time.
    })
}

/// A motion vector in quarter-pel units (the H.264 storage unit).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionVector {
    pub mv_x: i16,
    pub mv_y: i16,
}

impl MotionVector {
    pub const fn new(x: i16, y: i16) -> Self {
        Self { mv_x: x, mv_y: y }
    }
}

/// P_8x8 sub-macroblock type from spec Table 7-17 (list-0 only for Baseline).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PSubPartition {
    /// P_L0_8x8 — one 8×8 partition, one MV.
    P8x8,
    /// P_L0_8x4 — two horizontal 8×4 partitions, two MVs.
    P8x4,
    /// P_L0_4x8 — two vertical 4×8 partitions, two MVs.
    P4x8,
    /// P_L0_4x4 — four 4×4 partitions, four MVs.
    P4x4,
}

impl PSubPartition {
    pub fn from_code(code: u32) -> Option<Self> {
        Some(match code {
            0 => Self::P8x8,
            1 => Self::P8x4,
            2 => Self::P4x8,
            3 => Self::P4x4,
            _ => return None,
        })
    }

    /// Number of (mvd_x, mvd_y) pairs this sub-partition contributes
    /// (1 / 2 / 2 / 4 per spec).
    pub fn num_mvds(self) -> usize {
        match self {
            Self::P8x8 => 1,
            Self::P8x4 | Self::P4x8 => 2,
            Self::P4x4 => 4,
        }
    }
}

/// Motion vector field for a single P-slice MB. Layout is per-4×4 block in
/// raster order: `mvs[by * 4 + bx]` is the MV for the 4×4 block at local
/// coordinates `(bx, by)` within the MB.
///
/// When a partition spans multiple 4×4 blocks (e.g. P_16x16 covers all 16),
/// every block in the partition gets the same MV and ref_idx.
#[derive(Debug, Clone, Default)]
pub struct MvField {
    pub mvs: [MotionVector; 16],
    pub ref_idx: [i8; 16],
}


/// Frame-wide 4×4-granular motion vector grid. Used by the median predictor
/// to look up neighbour MVs across MB boundaries.
///
/// `ref_idx = -1` marks blocks that are intra-coded (or off-frame): the
/// predictor treats those as "not available" and routes around them.
pub struct MvPredictorContext {
    width_in_4x4: usize,
    height_in_4x4: usize,
    mv_grid: Vec<MotionVector>,
    ref_idx_grid: Vec<i8>,
}

impl MvPredictorContext {
    pub fn new(width_in_mbs: u32, height_in_mbs: u32) -> Self {
        let width_in_4x4 = (width_in_mbs * 4) as usize;
        let height_in_4x4 = (height_in_mbs * 4) as usize;
        let total = width_in_4x4 * height_in_4x4;
        Self {
            width_in_4x4,
            height_in_4x4,
            mv_grid: vec![MotionVector::default(); total],
            ref_idx_grid: vec![-1; total],
        }
    }

    pub fn width_in_4x4(&self) -> usize {
        self.width_in_4x4
    }

    pub fn set(&mut self, block_x: usize, block_y: usize, mv: MotionVector, ref_idx: i8) {
        if block_x >= self.width_in_4x4 || block_y >= self.height_in_4x4 {
            return;
        }
        let idx = block_y * self.width_in_4x4 + block_x;
        self.mv_grid[idx] = mv;
        self.ref_idx_grid[idx] = ref_idx;
    }

    pub fn get(&self, block_x: isize, block_y: isize) -> Option<(MotionVector, i8)> {
        if block_x < 0
            || block_y < 0
            || (block_x as usize) >= self.width_in_4x4
            || (block_y as usize) >= self.height_in_4x4
        {
            return None;
        }
        let idx = (block_y as usize) * self.width_in_4x4 + block_x as usize;
        let rid = self.ref_idx_grid[idx];
        if rid < 0 {
            None
        } else {
            Some((self.mv_grid[idx], rid))
        }
    }
}

/// Full AMVP predictor per H.264 spec § 8.4.1.3 / § 8.4.1.3.1. Handles
/// the P_16x8 / P_8x16 "directional" shortcuts before falling through to
/// the general median rule.
///
/// Spec § 8.4.1.3.1 (directional shortcuts):
/// * P_16x8 partition 0 (top):    if refB == curRef → mvB
/// * P_16x8 partition 1 (bottom): if refA == curRef → mvA
/// * P_8x16 partition 0 (left):   if refA == curRef → mvA
/// * P_8x16 partition 1 (right):  if refC == curRef → mvC
///
/// `part_w_4x4` / `part_h_4x4` / `mb_part_idx` describe the MB-level
/// partition (not sub-MB partitions within P_8x8). For P_8x8 sub-MB
/// partitions, pass the MB's 8×8 sub-partition dims (w=2, h=2) — the
/// shortcuts don't fire there, matching spec semantics where sub-MB
/// partitions always use the general median.
pub fn amvp_predict(
    left: Option<(MotionVector, i8)>,
    top: Option<(MotionVector, i8)>,
    top_right: Option<(MotionVector, i8)>,
    current_ref_idx: i8,
    part_w_4x4: usize,
    part_h_4x4: usize,
    mb_part_idx: u8,
) -> MotionVector {
    // Directional shortcuts (§ 8.4.1.3.1) — only for P_16x8 (4×2) and
    // P_8x16 (2×4) MB partitions.
    if part_w_4x4 == 4 && part_h_4x4 == 2 {
        // P_16x8: top partition uses top, bottom partition uses left.
        if mb_part_idx == 0 {
            if let Some((mv, r)) = top
                && r == current_ref_idx {
                    return mv;
                }
        } else if let Some((mv, r)) = left
        && r == current_ref_idx {
            return mv;
        }
    } else if part_w_4x4 == 2 && part_h_4x4 == 4 {
        // P_8x16: left partition uses left, right partition uses top-right.
        if mb_part_idx == 0 {
            if let Some((mv, r)) = left
                && r == current_ref_idx {
                    return mv;
                }
        } else if let Some((mv, r)) = top_right
        && r == current_ref_idx {
            return mv;
        }
    }
    median_mv(left, top, top_right, current_ref_idx)
}

/// Three-tap median filter that ignores unavailable neighbours per H.264
/// spec 8.4.1.3.
///
/// Rules:
/// * If exactly one neighbour has a matching `ref_idx`, use its MV.
/// * Otherwise, compute the componentwise median of the three (unavailable
///   neighbours are treated as zero MVs for the median's sake).
/// * If only one neighbour exists at all (e.g. top edge of frame), use it
///   directly (spec special case).
///
/// Note: the § 8.4.1.3.1 P_16x8 / P_8x16 directional shortcuts are
/// handled by [`amvp_predict`] BEFORE this function is called. This
/// function only implements the general § 8.4.1.3 median rule.
pub fn median_mv(
    left: Option<(MotionVector, i8)>,
    top: Option<(MotionVector, i8)>,
    top_right: Option<(MotionVector, i8)>,
    current_ref_idx: i8,
) -> MotionVector {
    // Single-neighbour special case: when only one of the three is available
    // at all, that one is the predictor (spec 8.4.1.3).
    let availability = [left.is_some(), top.is_some(), top_right.is_some()];
    let avail_count: u8 = availability.iter().map(|&b| b as u8).sum();
    if avail_count == 1
        && let Some((mv, _)) = left.or(top).or(top_right) {
            return mv;
        }

    // Single-matching-ref special case.
    let matching: Vec<Option<MotionVector>> = [left, top, top_right]
        .iter()
        .map(|n| {
            n.and_then(|(mv, rid)| {
                if rid == current_ref_idx {
                    Some(mv)
                } else {
                    None
                }
            })
        })
        .collect();
    let match_count: usize = matching.iter().filter(|m| m.is_some()).count();
    if match_count == 1
        && let Some(mv) = matching.iter().flatten().next() {
            return *mv;
        }

    // General case: componentwise median over the three neighbour MVs
    // (unavailable ones contribute zero per the spec's handling).
    let l = left.map(|(mv, _)| mv).unwrap_or_default();
    let t = top.map(|(mv, _)| mv).unwrap_or_default();
    let tr = top_right.map(|(mv, _)| mv).unwrap_or_default();

    MotionVector {
        mv_x: median3(l.mv_x, t.mv_x, tr.mv_x),
        mv_y: median3(l.mv_y, t.mv_y, tr.mv_y),
    }
}

#[inline]
fn median3(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.max(c)).min(b.max(c))
}

/// Read one signed Exp-Golomb MVD and, if it is non-zero, push its
/// suffix-LSB position into `mvd_positions`. Phase 3 plumbing helper.
#[inline]
fn read_mvd_capturing(
    reader: &mut RbspReader<'_>,
    ep_map: &EpByteMap,
    raw_data: &[u8],
    mvd_positions: &mut Vec<EmbeddablePosition>,
) -> Result<i16, H264Error> {
    let bits_before = reader.bits_read();
    let mvd = reader.read_se()? as i16;
    let bits_after = reader.bits_read();
    if let Some(p) = capture_mvd_position(bits_before, bits_after, mvd, ep_map, raw_data) {
        mvd_positions.push(p);
    }
    Ok(mvd)
}

/// Parse the MV / ref_idx / MVD syntax for a single P-slice MB and return
/// the resolved `MvField`. Updates `ctx` so subsequent MBs can use the
/// stored MVs for their own predictor lookups.
///
/// Supports `MbType::P16x16`, `P16x8`, `P8x16`, `P8x8`, and `P8x8ref0`.
/// Intra / skip / PCM types return `None`.
///
/// Phase 3a: each non-zero MVD contributes one `EmbeddablePosition`
/// (`EmbedDomain::MvdLsb`) pushed into `mvd_positions`.
pub fn parse_mv_field(
    reader: &mut RbspReader<'_>,
    mb_type: MbType,
    mb_x: u32,
    mb_y: u32,
    num_ref_idx_l0_active: u8,
    ctx: &mut MvPredictorContext,
    ep_map: &EpByteMap,
    raw_data: &[u8],
    mvd_positions: &mut Vec<EmbeddablePosition>,
) -> Result<Option<MvField>, H264Error> {
    let max_ref = num_ref_idx_l0_active.saturating_sub(1) as u32;
    let base_x = (mb_x * 4) as usize;
    let base_y = (mb_y * 4) as usize;

    let mut field = MvField::default();

    // Partition layouts in 4x4-block units:
    //   P_16x16: 1 partition  (bx=0, by=0, w=4, h=4)
    //   P_16x8 : 2 partitions (top then bottom half, each 4x2)
    //   P_8x16 : 2 partitions (left then right half, each 2x4)
    //   P_8x8  : 4 sub-MBs of size 2x2, each with its own sub_mb_type
    match mb_type {
        MbType::P16x16 => {
            let ref_idx = if max_ref > 0 {
                reader.read_te(max_ref)? as i8
            } else {
                0
            };
            let mvd_x = read_mvd_capturing(reader, ep_map, raw_data, mvd_positions)?;
            let mvd_y = read_mvd_capturing(reader, ep_map, raw_data, mvd_positions)?;
            resolve_partition(
                &mut field,
                ctx,
                base_x,
                base_y,
                0,
                0,
                4,
                4,
                4,
                4,
                0,
                ref_idx,
                (mvd_x, mvd_y),
            );
        }
        MbType::P16x8 => {
            // Two horizontal partitions: top half then bottom half.
            let mut ref_idxs = [0i8; 2];
            for r in ref_idxs.iter_mut() {
                *r = if max_ref > 0 {
                    reader.read_te(max_ref)? as i8
                } else {
                    0
                };
            }
            for (i, r) in ref_idxs.iter().enumerate() {
                let mvd_x = read_mvd_capturing(reader, ep_map, raw_data, mvd_positions)?;
                let mvd_y = read_mvd_capturing(reader, ep_map, raw_data, mvd_positions)?;
                let (off_y, h) = (i * 2, 2);
                resolve_partition(
                    &mut field,
                    ctx,
                    base_x,
                    base_y,
                    0,
                    off_y,
                    4,
                    h,
                    4,
                    2,
                    i as u8,
                    *r,
                    (mvd_x, mvd_y),
                );
            }
        }
        MbType::P8x16 => {
            let mut ref_idxs = [0i8; 2];
            for r in ref_idxs.iter_mut() {
                *r = if max_ref > 0 {
                    reader.read_te(max_ref)? as i8
                } else {
                    0
                };
            }
            for (i, r) in ref_idxs.iter().enumerate() {
                let mvd_x = read_mvd_capturing(reader, ep_map, raw_data, mvd_positions)?;
                let mvd_y = read_mvd_capturing(reader, ep_map, raw_data, mvd_positions)?;
                let (off_x, w) = (i * 2, 2);
                resolve_partition(
                    &mut field,
                    ctx,
                    base_x,
                    base_y,
                    off_x,
                    0,
                    w,
                    4,
                    2,
                    4,
                    i as u8,
                    *r,
                    (mvd_x, mvd_y),
                );
            }
        }
        MbType::P8x8 | MbType::P8x8ref0 => {
            // Spec order for P_8x8: 4 sub_mb_types, then 4 ref_idxs (or
            // implicit 0 for P8x8ref0), then per-sub-partition MVDs.
            let mut subs = [PSubPartition::P8x8; 4];
            for s in subs.iter_mut() {
                let code = reader.read_ue()?;
                *s = PSubPartition::from_code(code).ok_or_else(|| {
                    H264Error::CavlcError(format!("invalid P-slice sub_mb_type: {code}"))
                })?;
            }
            let mut ref_idxs = [0i8; 4];
            if mb_type != MbType::P8x8ref0 && max_ref > 0 {
                for r in ref_idxs.iter_mut() {
                    *r = reader.read_te(max_ref)? as i8;
                }
            }
            // Each of the 4 sub-MBs occupies a 2x2 block region in 4x4 units:
            //   sub0 (top-left): off_x=0, off_y=0
            //   sub1 (top-right): off_x=2, off_y=0
            //   sub2 (bottom-left): off_x=0, off_y=2
            //   sub3 (bottom-right): off_x=2, off_y=2
            let sub_origins = [(0usize, 0usize), (2, 0), (0, 2), (2, 2)];
            for i in 0..4 {
                let sub = subs[i];
                let (off_x, off_y) = sub_origins[i];
                // Partitions inside the sub-MB (in 4x4-block units):
                //   P8x8 -> 1 partition of size (2,2)
                //   P8x4 -> 2 partitions of size (2,1), stacked vertically
                //   P4x8 -> 2 partitions of size (1,2), side by side
                //   P4x4 -> 4 partitions of size (1,1)
                let parts: &[(usize, usize, usize, usize)] = match sub {
                    PSubPartition::P8x8 => &[(0, 0, 2, 2)],
                    PSubPartition::P8x4 => &[(0, 0, 2, 1), (0, 1, 2, 1)],
                    PSubPartition::P4x8 => &[(0, 0, 1, 2), (1, 0, 1, 2)],
                    PSubPartition::P4x4 => {
                        &[(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)]
                    }
                };
                for &(dx, dy, pw, ph) in parts {
                    let mvd_x = read_mvd_capturing(reader, ep_map, raw_data, mvd_positions)?;
                    let mvd_y = read_mvd_capturing(reader, ep_map, raw_data, mvd_positions)?;
                    // P_8x8: the MB-level partition is 8×8 (w=2, h=2 in
                    // 4×4 units). Sub-MB sub-partitions (8×4, 4×8, 4×4)
                    // don't trigger § 8.4.1.3.1 directional shortcuts —
                    // pass the 8×8 MB-partition dims, which falls
                    // through to general median.
                    resolve_partition(
                        &mut field,
                        ctx,
                        base_x,
                        base_y,
                        off_x + dx,
                        off_y + dy,
                        pw,
                        ph,
                        2,
                        2,
                        i as u8,
                        ref_idxs[i],
                        (mvd_x, mvd_y),
                    );
                }
            }
        }
        _ => return Ok(None),
    }

    Ok(Some(field))
}

/// Resolve a single partition: compute the predictor, add the MVD, write
/// the absolute MV into every 4×4 block in the partition region, and
/// publish to the predictor context.
///
/// `mb_part_w_4x4` / `mb_part_h_4x4` / `mb_part_idx` describe the
/// MB-level partition for § 8.4.1.3.1 directional-shortcut eligibility.
/// For P_8x8 sub-MB partitions, pass the 8×8 sub-MB dims (w=2, h=2) so
/// the shortcuts don't fire.
#[allow(clippy::too_many_arguments)]
fn resolve_partition(
    field: &mut MvField,
    ctx: &mut MvPredictorContext,
    base_x: usize,
    base_y: usize,
    off_x: usize,
    off_y: usize,
    width: usize,
    height: usize,
    mb_part_w_4x4: usize,
    mb_part_h_4x4: usize,
    mb_part_idx: u8,
    ref_idx: i8,
    mvd: (i16, i16),
) {
    // Neighbour lookup uses the partition's top-left 4×4 block as the
    // anchor. Top-right is the 4×4 block immediately right of the
    // partition's top-right corner on the same row (per spec 6.4.11.7).
    let top_left_x = (base_x + off_x) as isize;
    let top_left_y = (base_y + off_y) as isize;

    let a = ctx.get(top_left_x - 1, top_left_y); // left
    let b = ctx.get(top_left_x, top_left_y - 1); // top
    let c_x = top_left_x + width as isize;
    let c_y = top_left_y - 1;
    // Spec fallback: if C unavailable, use D (top-left diagonal).
    let c = ctx.get(c_x, c_y).or_else(|| ctx.get(top_left_x - 1, top_left_y - 1));

    let mvp = amvp_predict(a, b, c, ref_idx, mb_part_w_4x4, mb_part_h_4x4, mb_part_idx);
    let mv = MotionVector {
        mv_x: mvp.mv_x.wrapping_add(mvd.0),
        mv_y: mvp.mv_y.wrapping_add(mvd.1),
    };

    // Fill the partition region in the MB field and the frame-wide grid.
    for dy in 0..height {
        for dx in 0..width {
            let block_x_in_mb = off_x + dx;
            let block_y_in_mb = off_y + dy;
            let idx_in_mb = block_y_in_mb * 4 + block_x_in_mb;
            if idx_in_mb < 16 {
                field.mvs[idx_in_mb] = mv;
                field.ref_idx[idx_in_mb] = ref_idx;
            }
            ctx.set(base_x + off_x + dx, base_y + off_y + dy, mv, ref_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median3_picks_the_middle_of_three() {
        assert_eq!(median3(1, 2, 3), 2);
        assert_eq!(median3(3, 2, 1), 2);
        assert_eq!(median3(5, 5, 1), 5);
        assert_eq!(median3(-10, 0, 5), 0);
    }

    #[test]
    fn median_mv_single_neighbour_uses_it_directly() {
        // Only left exists → predictor = left.mv.
        let left = Some((MotionVector::new(8, -4), 0));
        let pred = median_mv(left, None, None, 0);
        assert_eq!(pred, MotionVector::new(8, -4));

        // Only top exists → predictor = top.mv.
        let top = Some((MotionVector::new(2, 2), 0));
        let pred = median_mv(None, top, None, 0);
        assert_eq!(pred, MotionVector::new(2, 2));
    }

    #[test]
    fn median_mv_single_match_on_refidx() {
        // Three neighbours, one matches ref_idx=1, other two don't. Predictor
        // should be the matching neighbour.
        let left = Some((MotionVector::new(100, 100), 0));
        let top = Some((MotionVector::new(50, 50), 1));
        let top_right = Some((MotionVector::new(0, 0), 0));
        let pred = median_mv(left, top, top_right, 1);
        assert_eq!(pred, MotionVector::new(50, 50));
    }

    #[test]
    fn median_mv_three_neighbours_componentwise_median() {
        // All three match ref_idx=0. Expect componentwise median.
        let left = Some((MotionVector::new(1, 10), 0));
        let top = Some((MotionVector::new(2, 20), 0));
        let top_right = Some((MotionVector::new(3, 30), 0));
        let pred = median_mv(left, top, top_right, 0);
        assert_eq!(pred, MotionVector::new(2, 20));
    }

    #[test]
    fn mv_predictor_context_roundtrips_writes() {
        let mut ctx = MvPredictorContext::new(4, 4); // 16x16 4x4-block grid
        ctx.set(3, 7, MotionVector::new(42, -7), 1);
        assert_eq!(ctx.get(3, 7), Some((MotionVector::new(42, -7), 1)));
        // Intra / uninitialised block should return None.
        assert_eq!(ctx.get(0, 0), None);
        // Out-of-bounds should also return None.
        assert_eq!(ctx.get(-1, 0), None);
        assert_eq!(ctx.get(0, -1), None);
        assert_eq!(ctx.get(100, 100), None);
    }

    // ---- Phase 3a: MVD position capture tests --------------------------

    /// Build a minimal `EpByteMap` that assumes no emulation-prevention bytes
    /// (rbsp_byte == raw_byte) for `n` bytes.
    fn identity_ep_map(n: usize) -> EpByteMap {
        EpByteMap {
            rbsp_to_raw: (0..n).collect(),
        }
    }

    #[test]
    fn capture_mvd_position_skips_zero_codeword() {
        let ep_map = identity_ep_map(4);
        let raw = [0u8; 4];
        // mvd=0 → codeNum=0, codeword is the single bit `1`, no suffix.
        // bits_before=0, bits_after=1, len=1. Must return None.
        let p = capture_mvd_position(0, 1, 0, &ep_map, &raw);
        assert!(p.is_none(), "mvd=0 must produce no embeddable position");
    }

    #[test]
    fn capture_mvd_position_marks_suffix_lsb_for_nonzero_mvd() {
        let ep_map = identity_ep_map(8);
        let raw = [0u8; 8];
        // mvd=1 → codeNum=1 → codeword `010` (3 bits), lz=1. Last bit is the
        // suffix LSB. bits_before=0, bits_after=3.
        let p =
            capture_mvd_position(0, 3, 1, &ep_map, &raw).expect("non-zero mvd must capture");
        assert_eq!(p.domain, EmbedDomain::MvdLsb);
        assert_eq!(p.block_idx, MVD_BLOCK_IDX_SENTINEL);
        assert_eq!(p.coeff_value, 1);
        assert_eq!(p.raw_byte_offset, 0);
        // LSB is the last bit — bit index 2 within byte 0.
        assert_eq!(p.bit_offset, 2);
    }

    #[test]
    fn capture_mvd_position_marks_large_mvd_at_long_suffix() {
        let ep_map = identity_ep_map(8);
        let raw = [0u8; 8];
        // mvd=3 → codeNum=5 → `00110` (5 bits), lz=2. Starting at bits_before=5
        // (byte 0 bit 5), bits_after = 10 → suffix LSB at bit index 9, which
        // is byte 1 bit 1.
        let p =
            capture_mvd_position(5, 10, 3, &ep_map, &raw).expect("non-zero mvd must capture");
        assert_eq!(p.raw_byte_offset, 1);
        assert_eq!(p.bit_offset, 1);
        assert_eq!(p.coeff_value, 3);
    }

    #[test]
    fn capture_mvd_position_preserves_negative_mvd_in_coeff_value() {
        let ep_map = identity_ep_map(4);
        let raw = [0u8; 4];
        // mvd=-2 → codeNum=4 → `00100` (5 bits), lz=2.
        let p = capture_mvd_position(0, 5, -2, &ep_map, &raw)
            .expect("non-zero mvd must capture");
        assert_eq!(p.coeff_value, -2);
        assert_eq!(p.domain, EmbedDomain::MvdLsb);
    }

    #[test]
    fn capture_mvd_position_rejects_malformed_length() {
        let ep_map = identity_ep_map(4);
        let raw = [0u8; 4];
        // Even-length codeword is impossible for Exp-Golomb — reject.
        assert!(capture_mvd_position(0, 4, 2, &ep_map, &raw).is_none());
        // Length < 3 with non-zero mvd is impossible (min mvd!=0 codeword = 3 bits).
        assert!(capture_mvd_position(0, 2, 2, &ep_map, &raw).is_none());
    }

    #[test]
    fn parse_mv_field_captures_mvd_positions_from_synthetic_p16x16() {
        use super::super::bitstream::{EpByteMap, RbspReader};
        // Synthesise a tiny P_16x16 payload with ref_idx implicit (max_ref=0)
        // and mvd_x=3, mvd_y=-2.
        //   Signed Exp-Golomb mapping: codeNum=5 -> +3, codeNum=4 -> -2.
        //   codeNum=5 -> codeword `00110` (lz=2, suffix binary 10).
        //   codeNum=4 -> codeword `00101` (lz=2, suffix binary 01).
        // Total 10 bits packed MSB-first:
        //   00110 00101 000000 = 0011 0001 0100 0000 = 0x31, 0x40.
        let bytes = [0x31u8, 0x40];
        let mut reader = RbspReader::new(&bytes);
        let ep_map = EpByteMap {
            rbsp_to_raw: vec![0, 1],
        };
        let mut ctx = MvPredictorContext::new(1, 1);
        let mut positions = Vec::new();
        let field = parse_mv_field(
            &mut reader,
            MbType::P16x16,
            0,
            0,
            1,
            &mut ctx,
            &ep_map,
            &bytes,
            &mut positions,
        )
        .expect("parse")
        .expect("p16x16 returns Some");
        // Both mvd_x and mvd_y are non-zero → two embeddable positions.
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].coeff_value, 3);
        assert_eq!(positions[1].coeff_value, -2);
        for p in &positions {
            assert_eq!(p.domain, EmbedDomain::MvdLsb);
            assert_eq!(p.block_idx, MVD_BLOCK_IDX_SENTINEL);
        }
        // Resolved MV should be (3, -2) with a zero predictor (no neighbours).
        assert_eq!(field.mvs[0], MotionVector::new(3, -2));
    }

    #[test]
    fn parse_mv_field_skips_zero_mvds_in_synthetic_p16x16() {
        use super::super::bitstream::{EpByteMap, RbspReader};
        // Both MVDs = 0 → each codeword is the single bit `1`. Two bits total,
        // packed MSB-first = `1100 0000` = 0xC0.
        let bytes = [0xC0u8];
        let mut reader = RbspReader::new(&bytes);
        let ep_map = EpByteMap {
            rbsp_to_raw: vec![0],
        };
        let mut ctx = MvPredictorContext::new(1, 1);
        let mut positions = Vec::new();
        parse_mv_field(
            &mut reader,
            MbType::P16x16,
            0,
            0,
            1,
            &mut ctx,
            &ep_map,
            &bytes,
            &mut positions,
        )
        .expect("parse");
        assert!(
            positions.is_empty(),
            "mvd=0 pairs must produce no embeddable positions"
        );
    }
}
