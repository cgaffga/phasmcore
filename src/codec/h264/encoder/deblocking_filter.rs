// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! In-loop deblocking filter. Phase 6A polish #1 (per H.264 § 8.7).
//!
//! The filter operates on the reconstructed frame after all MBs have
//! been decoded. For each 4×4-block edge (4 vertical + 4 horizontal
//! edges per MB), a boundary strength `bs` ∈ [0, 4] is derived; `bs`
//! plus the per-side QP determines whether to filter at all and
//! which tap length to use.
//!
//! This encoder applies the filter as a frame-level post-process so
//! that the encoder's DPB matches what the decoder reconstructs — a
//! prerequisite for drift-free inter prediction.

use super::motion_estimation::MotionVector;
use super::partition_state::EncoderMvGrid;
use super::reconstruction::ReconBuffer;
use crate::codec::h264::transform::derive_chroma_qp;

// ─── Spec tables ─────────────────────────────────────────────────
//
// Alpha and Beta thresholds per spec Table 8-16. Indexed by the
// `indexA` / `indexB` values derived from `filterOffsetA` / `filterOffsetB`
// and the per-edge QP. Our encoder always uses offset = 0, so these
// are indexed directly by `qp`.

const ALPHA_TABLE: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
    22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226,
    255, 255,
];

const BETA_TABLE: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8,
    8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

/// Spec Table 8-17 `tC0` — clip threshold indexed by `bs ∈ {1, 2, 3}`
/// (row) and `indexA = qp + filterOffsetA` (column, 0..=51). `bs = 4`
/// doesn't use tC0 (it uses the strong filter from spec § 8.7.2.4).
///
/// Values transcribed directly from spec Table 8-17 "Value of
/// variable t′C0 as a function of indexA and bS" (ITU-T H.264
/// 03/2010, page 207). Tabular numerical data is defined verbatim
/// by the standard — the table is a hand-tuned deblocking-strength
/// schedule chosen by the H.264 committee, not derived from a
/// closed-form formula.
///
/// Historical note (2026-04-20): prior versions of this table had
/// values shifted by ~10 indices low versus spec, producing ~13%
/// pixel-level enc-vs-dec drift per frame (MSE 0.14, PSNR 57 dB).
/// That was the root cause of a "video is totally destroyed"
/// visual bug that persisted after all per-MB-QP + bs=0/1
/// inter-edge structural fixes. `tc0_table_matches_spec` unit test
/// below locks every cell against the spec.
const TC0_TABLE: [[u8; 52]; 3] = [
    // bs = 1
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13,
    ],
    // bs = 2
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 10, 11, 12, 13, 15, 17,
    ],
    // bs = 3
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 23, 25,
    ],
];

#[inline]
fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.clamp(lo, hi)
}

#[inline]
fn clip1_y(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

// ─── Boundary strength ───────────────────────────────────────────
//
// For our encoder (I-only currently; P with inter partitions in
// Phase 6B):
//   - bs = 4: edges of an I_NxN MB at its outer 16×16 boundary, OR
//             both sides intra-coded AND edge is an MB boundary.
//   - bs = 3: at least one side intra-coded (not at MB boundary is
//             bs=3 too; simplification treats internal intra edges
//             as bs=3).
//   - bs = 2: at least one side has coded residual (nonzero coeffs).
//   - bs = 1: MVs differ > 4 qpel or reference frames differ (P-only).
//   - bs = 0: neither side coded, same ref, MV diff < 4 qpel.

/// Which kind of edge are we filtering: an MB-boundary edge (crosses
/// between two MBs) or an internal 4×4-block edge (within one MB).
#[derive(Debug, Clone, Copy)]
pub enum EdgeKind {
    MbBoundary,
    Internal,
}

/// Compute the boundary strength for one edge between sides `p` (left
/// or top) and `q` (right or bottom), both 4×4 block positions in
/// absolute 4×4-block coordinates.
///
/// Per spec § 8.7.2.1:
/// - bs=4: edge at MB boundary AND either side is intra.
/// - bs=3: either side is intra (not at MB boundary → internal).
/// - bs=2: either side has nonzero transform coefficients.
/// - bs=1: both sides inter AND (different refs OR |mv_diff|>=4 qpel
///   on either axis).
/// - bs=0: both sides inter AND same ref AND |mv_diff|<4 on both axes.
///
/// `p_intra` / `q_intra`: true if the 4×4 block belongs to an intra MB.
/// `p_has_coeff` / `q_has_coeff`: true if the block has any nonzero AC
/// residual. `p_ref` / `q_ref`: ref-list index for each side
/// (REF_IDX_NONE=-1 for intra). `p_mv` / `q_mv`: quarter-pel MVs.
pub fn boundary_strength(
    edge_kind: EdgeKind,
    p_intra: bool,
    q_intra: bool,
    p_has_coeff: bool,
    q_has_coeff: bool,
    p_ref: i8,
    q_ref: i8,
    p_mv: MotionVector,
    q_mv: MotionVector,
) -> u8 {
    if p_intra || q_intra {
        match edge_kind {
            EdgeKind::MbBoundary => 4,
            EdgeKind::Internal => 3,
        }
    } else if p_has_coeff || q_has_coeff {
        2
    } else if p_ref != q_ref
        || (p_mv.mv_x as i32 - q_mv.mv_x as i32).abs() >= 4
        || (p_mv.mv_y as i32 - q_mv.mv_y as i32).abs() >= 4
    {
        1
    } else {
        // Both inter, same ref, MVs within ±3 qpel — smooth motion,
        // no filter (bs=0 leaves pixels untouched).
        0
    }
}

// ─── Filter4: bs = 4 strong filter (luma) ─────────────────────────
//
// Spec § 8.7.2.2. Modifies up to 3 pixels on each side.

fn filter_luma_bs4(p: [i32; 4], q: [i32; 4], alpha: i32, beta: i32) -> ([i32; 3], [i32; 3]) {
    // p[0] = closest to edge, p[3] farthest.
    let [p0, p1, p2, p3] = p;
    let [q0, q1, q2, q3] = q;

    let mut new_p = [p0, p1, p2];
    let mut new_q = [q0, q1, q2];

    let ap = (p2 - p0).abs();
    let aq = (q2 - q0).abs();

    let filter_p_strong = ap < beta && (p0 - q0).abs() < ((alpha >> 2) + 2);
    let filter_q_strong = aq < beta && (p0 - q0).abs() < ((alpha >> 2) + 2);

    if filter_p_strong {
        new_p[0] = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3;
        new_p[1] = (p2 + p1 + p0 + q0 + 2) >> 2;
        new_p[2] = (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3;
    } else {
        new_p[0] = (2 * p1 + p0 + q1 + 2) >> 2;
    }

    if filter_q_strong {
        new_q[0] = (q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3;
        new_q[1] = (q2 + q1 + q0 + p0 + 2) >> 2;
        new_q[2] = (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3;
    } else {
        new_q[0] = (2 * q1 + q0 + p1 + 2) >> 2;
    }

    (new_p, new_q)
}

// ─── Filter2: bs = 1..3 normal filter (luma) ──────────────────────
//
// Spec § 8.7.2.3. Modifies up to 2 pixels on each side.

fn filter_luma_bs_lt4(
    p: [i32; 4],
    q: [i32; 4],
    _alpha: i32,
    beta: i32,
    tc0: i32,
) -> ([i32; 3], [i32; 3]) {
    let [p0, p1, p2, _p3] = p;
    let [q0, q1, q2, _q3] = q;

    let mut new_p = [p0, p1, p2];
    let mut new_q = [q0, q1, q2];

    let ap = (p2 - p0).abs();
    let aq = (q2 - q0).abs();

    let tc = tc0 + (if ap < beta { 1 } else { 0 }) + (if aq < beta { 1 } else { 0 });

    let delta = clip3(
        -tc,
        tc,
        (((q0 - p0) << 2) + (p1 - q1) + 4) >> 3,
    );
    new_p[0] = clip1_y(p0 + delta) as i32;
    new_q[0] = clip1_y(q0 - delta) as i32;

    if ap < beta {
        new_p[1] = p1 + clip3(-tc0, tc0, (p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1);
    }
    if aq < beta {
        new_q[1] = q1 + clip3(-tc0, tc0, (q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1);
    }

    (new_p, new_q)
}

// ─── Chroma filters ──────────────────────────────────────────────
//
// Chroma uses the same filter logic but only modifies p0, q0 (one
// pixel each side). Spec § 8.7.2.2 / 8.7.2.3.

fn filter_chroma_bs4(p: [i32; 2], q: [i32; 2]) -> (i32, i32) {
    let [p0, p1] = p;
    let [q0, q1] = q;
    let new_p0 = (2 * p1 + p0 + q1 + 2) >> 2;
    let new_q0 = (2 * q1 + q0 + p1 + 2) >> 2;
    (new_p0, new_q0)
}

fn filter_chroma_bs_lt4(p: [i32; 2], q: [i32; 2], tc0: i32) -> (i32, i32) {
    let [p0, p1] = p;
    let [q0, q1] = q;
    let tc = tc0 + 1;
    let delta = clip3(
        -tc,
        tc,
        (((q0 - p0) << 2) + (p1 - q1) + 4) >> 3,
    );
    let new_p0 = clip1_y(p0 + delta) as i32;
    let new_q0 = clip1_y(q0 - delta) as i32;
    (new_p0, new_q0)
}

// ─── Filter one luma edge ────────────────────────────────────────
//
// `p_samples` and `q_samples` are 4-pixel columns (for vertical
// edges) or rows (for horizontal edges) across the edge. Writes
// back the filtered values in place.

fn filter_luma_edge(p_samples: &mut [u8; 4], q_samples: &mut [u8; 4], bs: u8, qp: u8) {
    if bs == 0 {
        return;
    }
    let alpha = ALPHA_TABLE[qp as usize] as i32;
    let beta = BETA_TABLE[qp as usize] as i32;

    let p = [
        p_samples[0] as i32,
        p_samples[1] as i32,
        p_samples[2] as i32,
        p_samples[3] as i32,
    ];
    let q = [
        q_samples[0] as i32,
        q_samples[1] as i32,
        q_samples[2] as i32,
        q_samples[3] as i32,
    ];

    let p0 = p[0];
    let q0 = q[0];
    let p1 = p[1];
    let q1 = q[1];

    // Per § 8.7.2.1: skip filter if any of these gate conditions fail.
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return;
    }

    let (new_p, new_q) = if bs == 4 {
        filter_luma_bs4(p, q, alpha, beta)
    } else {
        let tc0 = TC0_TABLE[(bs - 1) as usize][qp as usize] as i32;
        filter_luma_bs_lt4(p, q, alpha, beta, tc0)
    };

    for i in 0..3 {
        p_samples[i] = clip1_y(new_p[i]);
        q_samples[i] = clip1_y(new_q[i]);
    }
}

fn filter_chroma_edge(p_samples: &mut [u8; 2], q_samples: &mut [u8; 2], bs: u8, qp: u8) {
    if bs == 0 {
        return;
    }
    let alpha = ALPHA_TABLE[qp as usize] as i32;
    let beta = BETA_TABLE[qp as usize] as i32;
    let p = [p_samples[0] as i32, p_samples[1] as i32];
    let q = [q_samples[0] as i32, q_samples[1] as i32];

    if (p[0] - q[0]).abs() >= alpha
        || (p[1] - p[0]).abs() >= beta
        || (q[1] - q[0]).abs() >= beta
    {
        return;
    }

    let (new_p0, new_q0) = if bs == 4 {
        filter_chroma_bs4(p, q)
    } else {
        let tc0 = TC0_TABLE[(bs - 1) as usize][qp as usize] as i32;
        filter_chroma_bs_lt4(p, q, tc0)
    };
    p_samples[0] = clip1_y(new_p0);
    q_samples[0] = clip1_y(new_q0);
}

// ─── Frame-level post-filter driver ──────────────────────────────

/// Apply the in-loop deblocking filter to the full reconstructed
/// frame. Runs after all MBs are reconstructed. Uses raster-order
/// per-MB filter application: for each MB in order, filter its 4
/// vertical edges (left→right), then its 4 horizontal edges
/// (top→bottom).
///
/// Per-edge parameters are derived from:
///   - `qp_grid[mb_y * mb_w + mb_x]`: the actual QP the encoder used
///     for that MB (may differ from slice QP due to AQ). The filter
///     looks up alpha/beta/tc0 using `qp_avg = (qp_p + qp_q + 1) >> 1`
///     per spec § 8.7.2.1.
///   - `intra_grid[mb_y * mb_w + mb_x]`: true if the MB was coded as
///     I_4x4 or I_16x16 (including the intra-in-P fallback path).
///     bs=4 at MB boundary when either side is intra, bs=3 for
///     internal edges of intra MBs.
///   - `coded_flags[by * w4 + bx]`: per-4×4 block "has nonzero AC
///     coefficients" flag. Drives bs=2 vs bs=0/1 for inter edges.
pub fn filter_frame(
    recon: &mut ReconBuffer,
    qp_grid: &[u8],
    intra_grid: &[bool],
    coded_flags: &[bool],
    mv_grid: Option<&EncoderMvGrid>,
) {
    filter_frame_with_transform(recon, qp_grid, intra_grid, None, coded_flags, mv_grid);
}

/// Variant that knows per-MB `transform_8x8_flag`. When set for an MB,
/// the filter skips that MB's internal 4-pixel-grid edges per spec
/// § 8.7.2.1 (8×8-transform MBs keep edges only on the 8-pixel grid).
pub fn filter_frame_with_transform(
    recon: &mut ReconBuffer,
    qp_grid: &[u8],
    intra_grid: &[bool],
    transform_8x8_grid: Option<&[bool]>,
    coded_flags: &[bool],
    mv_grid: Option<&EncoderMvGrid>,
) {
    let mb_w = (recon.width / 16) as usize;
    let mb_h = (recon.height / 16) as usize;
    let w4 = mb_w * 4;
    assert_eq!(qp_grid.len(), mb_w * mb_h);
    assert_eq!(intra_grid.len(), mb_w * mb_h);
    if let Some(g) = transform_8x8_grid {
        assert_eq!(g.len(), mb_w * mb_h);
    }

    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            filter_mb_edges(
                recon, mb_x, mb_y, mb_w, w4, qp_grid, intra_grid,
                transform_8x8_grid, coded_flags, mv_grid,
            );
        }
    }
}

/// Fetch (ref_idx, mv) for the 4×4 block at absolute coords `(bx, by)`.
/// Returns (REF_IDX_NONE, ZERO) when mv_grid is absent (I-slice) or
/// the block is intra-coded.
#[inline]
fn mv_at(mv_grid: Option<&EncoderMvGrid>, bx: usize, by: usize) -> (i8, MotionVector) {
    use super::partition_state::REF_IDX_NONE;
    match mv_grid.and_then(|g| g.get(bx as isize, by as isize)) {
        Some((mv, r)) => (r, mv),
        None => (REF_IDX_NONE, MotionVector::ZERO),
    }
}

fn coded(coded_flags: &[bool], w4: usize, bx: usize, by: usize) -> bool {
    if bx >= w4 || by * w4 + bx >= coded_flags.len() {
        return false;
    }
    coded_flags[by * w4 + bx]
}

#[allow(clippy::too_many_arguments)]
fn filter_mb_edges(
    recon: &mut ReconBuffer,
    mb_x: usize,
    mb_y: usize,
    mb_w: usize,
    w4: usize,
    qp_grid: &[u8],
    intra_grid: &[bool],
    transform_8x8_grid: Option<&[bool]>,
    coded_flags: &[bool],
    mv_grid: Option<&EncoderMvGrid>,
) {
    let origin_x = mb_x * 16;
    let origin_y = mb_y * 16;
    let cur_idx = mb_y * mb_w + mb_x;
    let cur_qp = qp_grid[cur_idx];
    let cur_intra = intra_grid[cur_idx];
    // When this MB uses 8×8 transform, internal 4-pixel-grid edges
    // at (edge_x ∈ {4, 12}) and (edge_y ∈ {4, 12}) are not filtered.
    // MB-boundary (edge = 0) and 8-pixel-grid internal (edge = 8)
    // edges are always filtered per spec § 8.7.2.1.
    let cur_t8 = transform_8x8_grid
        .is_some_and(|g| g[cur_idx]);

    // ── Vertical edges: x = 0 (cross-MB), 4, 8, 12 (internal) ──
    for edge_x in [0usize, 4, 8, 12] {
        if edge_x == 0 && mb_x == 0 {
            continue; // frame boundary — nothing to filter against
        }
        // Skip internal 4-pixel-grid vertical edges inside 8×8-transformed MB.
        if cur_t8 && (edge_x == 4 || edge_x == 12) {
            continue;
        }
        let edge_kind = if edge_x == 0 {
            EdgeKind::MbBoundary
        } else {
            EdgeKind::Internal
        };
        // Per spec § 8.7.2.1: at a vertical edge, the "P" side is the
        // MB to the LEFT of the edge, the "Q" side is the MB to the
        // RIGHT. For internal edges both sides are the current MB.
        let (p_mb_idx, q_mb_idx) = if edge_x == 0 {
            (mb_y * mb_w + (mb_x - 1), cur_idx)
        } else {
            (cur_idx, cur_idx)
        };
        let p_qp = qp_grid[p_mb_idx];
        let q_qp = qp_grid[q_mb_idx];
        let p_intra = intra_grid[p_mb_idx];
        let q_intra = intra_grid[q_mb_idx];
        // qp_avg = (qp_p + qp_q + 1) >> 1 per spec § 8.7.2.1.
        let qp_avg = ((p_qp as u32 + q_qp as u32 + 1) >> 1) as u8;
        // chroma qp: average per-MB chroma QPs (derive_chroma_qp is
        // non-linear through Table 8-15 for qp ≥ 30 — average of
        // derived chroma QPs is NOT the same as derived-from-average).
        let p_qp_c = derive_chroma_qp(p_qp as i32, 0);
        let q_qp_c = derive_chroma_qp(q_qp as i32, 0);
        let qp_c_avg = ((p_qp_c + q_qp_c + 1) >> 1) as u8;

        // Luma: 4 4-sample columns across the edge, one per row of
        // 4×4 sub-blocks (rows 0, 4, 8, 12).
        for row4 in 0..4 {
            let p_bx = (origin_x + edge_x) / 4 - 1;
            let p_by = origin_y / 4 + row4;
            let q_bx = p_bx + 1;
            let q_by = p_by;
            let p_cof = coded(coded_flags, w4, p_bx, p_by);
            let q_cof = coded(coded_flags, w4, q_bx, q_by);
            let (p_ref, p_mv) = mv_at(mv_grid, p_bx, p_by);
            let (q_ref, q_mv) = mv_at(mv_grid, q_bx, q_by);
            let bs = boundary_strength(
                edge_kind, p_intra, q_intra, p_cof, q_cof, p_ref, q_ref, p_mv, q_mv,
            );
            for sample_row in 0..4 {
                let y = origin_y + row4 * 4 + sample_row;
                let x = origin_x + edge_x;
                let stride = recon.width as usize;
                let mut p = [
                    recon.y[y * stride + x - 4],
                    recon.y[y * stride + x - 3],
                    recon.y[y * stride + x - 2],
                    recon.y[y * stride + x - 1],
                ];
                let mut q = [
                    recon.y[y * stride + x],
                    recon.y[y * stride + x + 1],
                    recon.y[y * stride + x + 2],
                    recon.y[y * stride + x + 3],
                ];
                // Order for filter: p[0] closest to edge.
                let mut p_rev = [p[3], p[2], p[1], p[0]];
                filter_luma_edge(&mut p_rev, &mut q, bs, qp_avg);
                p[3] = p_rev[0];
                p[2] = p_rev[1];
                p[1] = p_rev[2];
                p[0] = p_rev[3];
                for i in 0..4 {
                    recon.y[y * stride + x - 4 + i] = p[i];
                }
                for i in 0..4 {
                    recon.y[y * stride + x + i] = q[i];
                }
            }
        }
        // Chroma: only edges at x = 0 and x = 8 in luma coords (=
        // x = 0, 4 in chroma coords) — spec § 8.7.2.
        if edge_x == 0 || edge_x == 8 {
            for comp in 0..2u8 {
                let c_edge_x = edge_x / 2;
                let c_origin_x = origin_x / 2;
                let c_origin_y = origin_y / 2;
                for sample_row in 0..8 {
                    let y = c_origin_y + sample_row;
                    let x = c_origin_x + c_edge_x;
                    let stride = (recon.width / 2) as usize;
                    let plane = if comp == 0 { &mut recon.cb } else { &mut recon.cr };
                    let mut p = [plane[y * stride + x - 1], plane[y * stride + x - 2]];
                    let mut q = [plane[y * stride + x], plane[y * stride + x + 1]];
                    // Derive bs from the luma block just above.
                    let p_bx = (origin_x + edge_x) / 4 - 1;
                    let p_by = origin_y / 4 + (sample_row / 2);
                    let q_bx = p_bx + 1;
                    let q_by = p_by;
                    let (p_ref_c, p_mv_c) = mv_at(mv_grid, p_bx, p_by);
                    let (q_ref_c, q_mv_c) = mv_at(mv_grid, q_bx, q_by);
                    let bs = boundary_strength(
                        if edge_x == 0 { EdgeKind::MbBoundary } else { EdgeKind::Internal },
                        p_intra,
                        q_intra,
                        coded(coded_flags, w4, p_bx, p_by),
                        coded(coded_flags, w4, q_bx, q_by),
                        p_ref_c,
                        q_ref_c,
                        p_mv_c,
                        q_mv_c,
                    );
                    filter_chroma_edge(&mut p, &mut q, bs, qp_c_avg);
                    plane[y * stride + x - 1] = p[0];
                    plane[y * stride + x - 2] = p[1];
                    plane[y * stride + x] = q[0];
                    plane[y * stride + x + 1] = q[1];
                }
            }
        }
    }

    // ── Horizontal edges: y = 0 (cross-MB), 4, 8, 12 (internal) ──
    for edge_y in [0usize, 4, 8, 12] {
        if edge_y == 0 && mb_y == 0 {
            continue;
        }
        // Skip internal 4-pixel-grid horizontal edges inside 8×8-transformed MB.
        if cur_t8 && (edge_y == 4 || edge_y == 12) {
            continue;
        }
        let edge_kind = if edge_y == 0 {
            EdgeKind::MbBoundary
        } else {
            EdgeKind::Internal
        };
        // P side is the MB ABOVE the edge, Q is below (current).
        let (p_mb_idx, q_mb_idx) = if edge_y == 0 {
            ((mb_y - 1) * mb_w + mb_x, cur_idx)
        } else {
            (cur_idx, cur_idx)
        };
        let p_qp = qp_grid[p_mb_idx];
        let q_qp = qp_grid[q_mb_idx];
        let p_intra = intra_grid[p_mb_idx];
        let q_intra = intra_grid[q_mb_idx];
        let qp_avg = ((p_qp as u32 + q_qp as u32 + 1) >> 1) as u8;
        let p_qp_c = derive_chroma_qp(p_qp as i32, 0);
        let q_qp_c = derive_chroma_qp(q_qp as i32, 0);
        let qp_c_avg = ((p_qp_c + q_qp_c + 1) >> 1) as u8;

        for col4 in 0..4 {
            let p_bx = origin_x / 4 + col4;
            let p_by = (origin_y + edge_y) / 4 - 1;
            let q_bx = p_bx;
            let q_by = p_by + 1;
            let (p_ref, p_mv) = mv_at(mv_grid, p_bx, p_by);
            let (q_ref, q_mv) = mv_at(mv_grid, q_bx, q_by);
            let bs = boundary_strength(
                edge_kind,
                p_intra,
                q_intra,
                coded(coded_flags, w4, p_bx, p_by),
                coded(coded_flags, w4, q_bx, q_by),
                p_ref,
                q_ref,
                p_mv,
                q_mv,
            );
            for sample_col in 0..4 {
                let x = origin_x + col4 * 4 + sample_col;
                let y = origin_y + edge_y;
                let stride = recon.width as usize;
                let mut p_rev = [
                    recon.y[(y - 1) * stride + x],
                    recon.y[(y - 2) * stride + x],
                    recon.y[(y - 3) * stride + x],
                    recon.y[(y - 4) * stride + x],
                ];
                let mut q = [
                    recon.y[y * stride + x],
                    recon.y[(y + 1) * stride + x],
                    recon.y[(y + 2) * stride + x],
                    recon.y[(y + 3) * stride + x],
                ];
                filter_luma_edge(&mut p_rev, &mut q, bs, qp_avg);
                recon.y[(y - 1) * stride + x] = p_rev[0];
                recon.y[(y - 2) * stride + x] = p_rev[1];
                recon.y[(y - 3) * stride + x] = p_rev[2];
                recon.y[(y - 4) * stride + x] = p_rev[3];
                for i in 0..4 {
                    recon.y[(y + i) * stride + x] = q[i];
                }
            }
        }
        if edge_y == 0 || edge_y == 8 {
            for comp in 0..2u8 {
                let c_edge_y = edge_y / 2;
                let c_origin_x = origin_x / 2;
                let c_origin_y = origin_y / 2;
                for sample_col in 0..8 {
                    let x = c_origin_x + sample_col;
                    let y = c_origin_y + c_edge_y;
                    let stride = (recon.width / 2) as usize;
                    let plane = if comp == 0 { &mut recon.cb } else { &mut recon.cr };
                    let mut p = [plane[(y - 1) * stride + x], plane[(y - 2) * stride + x]];
                    let mut q = [plane[y * stride + x], plane[(y + 1) * stride + x]];
                    let p_bx = origin_x / 4 + (sample_col / 2);
                    let p_by = (origin_y + edge_y) / 4 - 1;
                    let q_bx = p_bx;
                    let q_by = p_by + 1;
                    let (p_ref_c, p_mv_c) = mv_at(mv_grid, p_bx, p_by);
                    let (q_ref_c, q_mv_c) = mv_at(mv_grid, q_bx, q_by);
                    let bs = boundary_strength(
                        if edge_y == 0 { EdgeKind::MbBoundary } else { EdgeKind::Internal },
                        p_intra,
                        q_intra,
                        coded(coded_flags, w4, p_bx, p_by),
                        coded(coded_flags, w4, q_bx, q_by),
                        p_ref_c,
                        q_ref_c,
                        p_mv_c,
                        q_mv_c,
                    );
                    filter_chroma_edge(&mut p, &mut q, bs, qp_c_avg);
                    plane[(y - 1) * stride + x] = p[0];
                    plane[(y - 2) * stride + x] = p[1];
                    plane[y * stride + x] = q[0];
                    plane[(y + 1) * stride + x] = q[1];
                }
            }
        }
    }
    // Keep cur_qp / cur_intra in scope for readability / future use.
    let _ = (cur_qp, cur_intra);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec Table 8-17 "Value of variable t′C0 as a function of
    /// indexA and bS" (ITU-T H.264 03/2010, page 207), encoded
    /// directly as run-length `(value, count)` pairs per bS row.
    /// Any drift between `TC0_TABLE` and this re-encoded spec table
    /// trips the `tc0_table_matches_spec` lock-in test.
    #[test]
    fn tc0_table_matches_spec() {
        // (bS, run-length-encoded row)
        //
        //   bS = 1 from the spec:
        //     indexA 0..22 -> 0,   23..32 -> 1,  33..36 -> 2,
        //     37..39  -> 3,  40..42  -> 4,  43 -> 5,
        //     44..45  -> 6,  46 -> 7, 47 -> 8, 48 -> 9,
        //     49 -> 10, 50 -> 11, 51 -> 13
        //
        //   bS = 2 from the spec:
        //     indexA 0..20 -> 0,   21..30 -> 1,  31..34 -> 2,
        //     35..37  -> 3,  38..39  -> 4,  40..41 -> 5,
        //     42 -> 6, 43 -> 7, 44..45 -> 8, 46 -> 10, 47 -> 11,
        //     48 -> 12, 49 -> 13, 50 -> 15, 51 -> 17
        //
        //   bS = 3 from the spec:
        //     indexA 0..16 -> 0,   17..26 -> 1,  27..30 -> 2,
        //     31..33  -> 3,  34..36  -> 4,  37 -> 5,
        //     38..39  -> 6,  40 -> 7, 41 -> 8, 42 -> 9,
        //     43 -> 10, 44 -> 11, 45 -> 13, 46 -> 14, 47 -> 16,
        //     48 -> 18, 49 -> 20, 50 -> 23, 51 -> 25
        let spec: [&[(u8, u8)]; 3] = [
            &[
                (0, 23), (1, 10), (2, 4), (3, 3), (4, 3), (5, 1),
                (6, 2), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1),
                (13, 1),
            ],
            &[
                (0, 21), (1, 10), (2, 4), (3, 3), (4, 2), (5, 2),
                (6, 1), (7, 1), (8, 2), (10, 1), (11, 1), (12, 1),
                (13, 1), (15, 1), (17, 1),
            ],
            &[
                (0, 17), (1, 10), (2, 4), (3, 3), (4, 3), (5, 1),
                (6, 2), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1),
                (13, 1), (14, 1), (16, 1), (18, 1), (20, 1), (23, 1),
                (25, 1),
            ],
        ];
        for (bs_row_idx, run_length) in spec.iter().enumerate() {
            let mut expanded = Vec::with_capacity(52);
            for &(value, count) in run_length.iter() {
                for _ in 0..count {
                    expanded.push(value);
                }
            }
            assert_eq!(expanded.len(), 52, "spec row {} has {} entries, expected 52",
                bs_row_idx, expanded.len());
            for (indexa, &spec_val) in expanded.iter().enumerate() {
                assert_eq!(
                    TC0_TABLE[bs_row_idx][indexa], spec_val,
                    "TC0_TABLE[bS={}][indexA={}] = {} but Table 8-17 gives {}",
                    bs_row_idx + 1, indexa, TC0_TABLE[bs_row_idx][indexa], spec_val,
                );
            }
        }
    }

    #[test]
    fn filter_zero_bs_is_noop() {
        let mut p = [100u8, 101, 102, 103];
        let mut q = [104u8, 105, 106, 107];
        filter_luma_edge(&mut p, &mut q, 0, 30);
        assert_eq!(p, [100, 101, 102, 103]);
        assert_eq!(q, [104, 105, 106, 107]);
    }

    #[test]
    fn filter_bs4_smoothes_moderate_step() {
        // Moderate step within alpha/beta thresholds so the filter
        // gate doesn't reject the edge as "real content".
        // At qp=30: alpha=22, beta=8. |p0-q0|=15 passes, |p1-p0|=0
        // passes, |q1-q0|=0 passes.
        let mut p = [100u8, 100, 100, 100];
        let mut q = [115u8, 115, 115, 115];
        filter_luma_edge(&mut p, &mut q, 4, 30);
        // Both sides should shift toward the midpoint.
        assert!(p[0] > 100, "p[0] should rise toward 107, got {}", p[0]);
        assert!(q[0] < 115, "q[0] should fall toward 107, got {}", q[0]);
    }

    #[test]
    fn filter_bs4_rejects_sharp_step_over_alpha() {
        // A huge step that exceeds alpha — spec § 8.7.2.1 gates the
        // filter, output should equal input (real content, not
        // blocking artifact).
        let mut p = [100u8, 100, 100, 100];
        let mut q = [200u8, 200, 200, 200];
        filter_luma_edge(&mut p, &mut q, 4, 30);
        assert_eq!(p, [100, 100, 100, 100]);
        assert_eq!(q, [200, 200, 200, 200]);
    }

    #[test]
    fn filter_bs0_ignores_bs_but_samples_under_threshold() {
        // bs = 1 with samples within alpha/beta thresholds — should
        // produce a modest smoothing.
        let mut p = [100u8, 100, 100, 100];
        let mut q = [102u8, 102, 102, 102];
        let p_before = p;
        let q_before = q;
        filter_luma_edge(&mut p, &mut q, 1, 30);
        // Change should be small (samples are close → filter nudges
        // gently or not at all).
        let dp = (p[0] as i32 - p_before[0] as i32).abs();
        let dq = (q[0] as i32 - q_before[0] as i32).abs();
        assert!(dp <= 2 && dq <= 2);
    }

    #[test]
    fn alpha_table_matches_spec_fixed_points() {
        // Spec § 8.7.2.1 Table 8-16: alpha(0..=15) = 0, alpha(20) = 7,
        // alpha(30) = 18 (approximately), alpha(51) = 255.
        assert_eq!(ALPHA_TABLE[0], 0);
        assert_eq!(ALPHA_TABLE[15], 0);
        assert_eq!(ALPHA_TABLE[51], 255);
    }

    #[test]
    fn beta_table_matches_spec_fixed_points() {
        assert_eq!(BETA_TABLE[0], 0);
        assert_eq!(BETA_TABLE[51], 18);
    }

    #[test]
    fn boundary_strength_both_intra_mb_edge_is_4() {
        assert_eq!(
            boundary_strength(
                EdgeKind::MbBoundary, true, true, false, false,
                -1, -1, MotionVector::ZERO, MotionVector::ZERO,
            ),
            4
        );
    }

    #[test]
    fn boundary_strength_both_intra_internal_is_3() {
        assert_eq!(
            boundary_strength(
                EdgeKind::Internal, true, true, false, false,
                -1, -1, MotionVector::ZERO, MotionVector::ZERO,
            ),
            3
        );
    }

    #[test]
    fn boundary_strength_nonzero_coeff_is_2() {
        assert_eq!(
            boundary_strength(
                EdgeKind::Internal, false, false, true, false,
                0, 0, MotionVector::ZERO, MotionVector::ZERO,
            ),
            2
        );
    }

    #[test]
    fn boundary_strength_inter_same_ref_small_mv_is_0() {
        // Both inter, same ref, no coeffs, MVs within ±3 qpel → bs=0.
        assert_eq!(
            boundary_strength(
                EdgeKind::Internal, false, false, false, false,
                0, 0,
                MotionVector { mv_x: 5, mv_y: 5 },
                MotionVector { mv_x: 7, mv_y: 6 },
            ),
            0
        );
    }

    #[test]
    fn boundary_strength_inter_mv_diff_4_is_1() {
        // Both inter, same ref, no coeffs, mv_x diff = 4 qpel → bs=1.
        assert_eq!(
            boundary_strength(
                EdgeKind::Internal, false, false, false, false,
                0, 0,
                MotionVector { mv_x: 0, mv_y: 0 },
                MotionVector { mv_x: 4, mv_y: 0 },
            ),
            1
        );
    }

    #[test]
    fn boundary_strength_inter_diff_ref_is_1() {
        // Different ref indices → bs=1.
        assert_eq!(
            boundary_strength(
                EdgeKind::Internal, false, false, false, false,
                0, 1, MotionVector::ZERO, MotionVector::ZERO,
            ),
            1
        );
    }

    // ─── Phase 100-H: skip-internal-4-pixel-edge under transform_8x8 ──
    //
    // Per spec § 8.7.2.1, when an MB's `transform_size_8x8_flag` is
    // set, only the MB-boundary edges (x = 0 / y = 0) and the
    // 8-pixel-grid internal edges (x = 8 / y = 8) are filtered. The
    // 4-pixel-grid internal edges at x ∈ {4, 12} and y ∈ {4, 12}
    // are skipped.

    /// Build a 16×16 single-MB recon buffer with a vertical "step"
    /// at `x = step_col` (left half = `lo`, right half = `hi`).
    /// Chroma is set to flat 128.
    fn build_single_mb_recon_with_v_step(lo: u8, hi: u8, step_col: usize) -> ReconBuffer {
        let mut recon = ReconBuffer::new(16, 16).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                recon.y[y * 16 + x] = if x < step_col { lo } else { hi };
            }
        }
        for i in 0..64 {
            recon.cb[i] = 128;
            recon.cr[i] = 128;
        }
        recon
    }

    /// Build a 16×16 single-MB recon buffer with a horizontal step
    /// at `y = step_row` (top half = `lo`, bottom half = `hi`).
    fn build_single_mb_recon_with_h_step(lo: u8, hi: u8, step_row: usize) -> ReconBuffer {
        let mut recon = ReconBuffer::new(16, 16).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                recon.y[y * 16 + x] = if y < step_row { lo } else { hi };
            }
        }
        for i in 0..64 {
            recon.cb[i] = 128;
            recon.cr[i] = 128;
        }
        recon
    }

    #[test]
    fn transform_8x8_skips_internal_vertical_4pixel_edge() {
        // Single intra MB at QP 26. A luma step at x = 4 should be
        // smoothed by the deblock filter when the MB uses 4×4
        // transform (bs = 3 internal intra edge) — and UNTOUCHED
        // when the MB uses the 8×8 transform (spec § 8.7.2.1 skip).
        let qp_grid = vec![26u8; 1];
        let intra_grid = vec![true; 1];
        let coded_flags = vec![false; 16]; // 4×4 blocks per MB = 16.

        let mut recon_4x4 = build_single_mb_recon_with_v_step(100, 110, 4);
        filter_frame_with_transform(
            &mut recon_4x4, &qp_grid, &intra_grid, Some(&[false]),
            &coded_flags, None,
        );

        let mut recon_8x8 = build_single_mb_recon_with_v_step(100, 110, 4);
        filter_frame_with_transform(
            &mut recon_8x8, &qp_grid, &intra_grid, Some(&[true]),
            &coded_flags, None,
        );

        // 8×8 path must leave the edge-adjacent columns EXACTLY as
        // set in the input: x ∈ {1, 2, 3} stay at 100; x ∈ {4, 5, 6}
        // stay at 110.
        for y in 0..16 {
            for x in [1, 2, 3] {
                assert_eq!(
                    recon_8x8.y[y * 16 + x], 100,
                    "8×8 path should skip x=4 internal edge; (y={y}, x={x}) changed"
                );
            }
            for x in [4, 5, 6] {
                assert_eq!(
                    recon_8x8.y[y * 16 + x], 110,
                    "8×8 path should skip x=4 internal edge; (y={y}, x={x}) changed"
                );
            }
        }

        // 4×4 path must smooth the edge — at least one column in
        // {1, 2, 3} or {4, 5, 6} must have changed relative to input.
        let four_changed = (0..16).any(|y| {
            (1..=3).any(|x| recon_4x4.y[y * 16 + x] != 100)
                || (4..=6).any(|x| recon_4x4.y[y * 16 + x] != 110)
        });
        assert!(
            four_changed,
            "4×4 path should deblock the x=4 internal edge, but no column changed"
        );
    }

    #[test]
    fn transform_8x8_skips_internal_horizontal_4pixel_edge() {
        // Symmetric test for the y = 4 horizontal internal edge.
        let qp_grid = vec![26u8; 1];
        let intra_grid = vec![true; 1];
        let coded_flags = vec![false; 16];

        let mut recon_4x4 = build_single_mb_recon_with_h_step(100, 110, 4);
        filter_frame_with_transform(
            &mut recon_4x4, &qp_grid, &intra_grid, Some(&[false]),
            &coded_flags, None,
        );

        let mut recon_8x8 = build_single_mb_recon_with_h_step(100, 110, 4);
        filter_frame_with_transform(
            &mut recon_8x8, &qp_grid, &intra_grid, Some(&[true]),
            &coded_flags, None,
        );

        for x in 0..16 {
            for y in [1, 2, 3] {
                assert_eq!(
                    recon_8x8.y[y * 16 + x], 100,
                    "8×8 path should skip y=4 internal edge; (y={y}, x={x}) changed"
                );
            }
            for y in [4, 5, 6] {
                assert_eq!(
                    recon_8x8.y[y * 16 + x], 110,
                    "8×8 path should skip y=4 internal edge; (y={y}, x={x}) changed"
                );
            }
        }
        let four_changed = (0..16).any(|x| {
            (1..=3).any(|y| recon_4x4.y[y * 16 + x] != 100)
                || (4..=6).any(|y| recon_4x4.y[y * 16 + x] != 110)
        });
        assert!(
            four_changed,
            "4×4 path should deblock the y=4 internal edge, but no row changed"
        );
    }

    #[test]
    fn transform_8x8_still_filters_8pixel_internal_edge() {
        // The 8-pixel-grid internal edge at x = 8 (or y = 8) is an
        // 8×8 block boundary — spec § 8.7.2.1 keeps these filtered
        // regardless of transform_size_8x8_flag. Step at x = 8 must
        // get deblocked even when the MB is 8×8-transformed.
        let qp_grid = vec![26u8; 1];
        let intra_grid = vec![true; 1];
        let coded_flags = vec![false; 16];

        let mut recon = build_single_mb_recon_with_v_step(100, 110, 8);
        filter_frame_with_transform(
            &mut recon, &qp_grid, &intra_grid, Some(&[true]),
            &coded_flags, None,
        );

        // Expect at least one column in {5, 6, 7} or {8, 9, 10} to
        // have changed from its input value.
        let eight_filtered = (0..16).any(|y| {
            (5..=7).any(|x| recon.y[y * 16 + x] != 100)
                || (8..=10).any(|x| recon.y[y * 16 + x] != 110)
        });
        assert!(
            eight_filtered,
            "x=8 internal 8-pixel-grid edge must still be filtered under 8×8 transform"
        );
    }

    #[test]
    fn legacy_filter_frame_treats_mbs_as_4x4() {
        // The `filter_frame` legacy wrapper must behave as if
        // transform_8x8_grid = None (all MBs treated as 4×4-
        // transformed). Step at x = 4 should get deblocked.
        let qp_grid = vec![26u8; 1];
        let intra_grid = vec![true; 1];
        let coded_flags = vec![false; 16];

        let mut recon = build_single_mb_recon_with_v_step(100, 110, 4);
        filter_frame(&mut recon, &qp_grid, &intra_grid, &coded_flags, None);

        let changed = (0..16).any(|y| {
            (1..=3).any(|x| recon.y[y * 16 + x] != 100)
                || (4..=6).any(|x| recon.y[y * 16 + x] != 110)
        });
        assert!(
            changed,
            "legacy filter_frame should apply deblock at x=4 internal edge"
        );
    }
}
