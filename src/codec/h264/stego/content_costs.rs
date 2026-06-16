// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Tier 3 content-adaptive cost computation for the unified video
//! stego pipeline.
//!
//! Bridges CABAC walker output ([`DomainCover`] in
//! [`crate::codec::h264::stego::inject`]) to the existing J-UNIWARD
//! wavelet-based cost machinery in
//! [`crate::stego::cost::h264_uniward`] (originally built for the
//! now-retired CAVLC video pipeline; reused unchanged by the current
//! OpenH264 stego path, which is unified onto this Scheme A cover).
//!
//! ## What this delivers
//!
//! Given the raw YUV input + the walker's 4-domain cover
//! ([`DomainCover`] with per-position [`PositionKey`]s), returns a
//! [`DomainCosts`] vector where each cost reflects local
//! detectability:
//!
//! - **Low cost** → textured / high-entropy region → hard to detect a
//!   flip → STC prefers to flip here.
//! - **High cost** → smooth / low-entropy region → easy to detect →
//!   STC avoids.
//! - **Infinite cost** (`f32::INFINITY`) → DC positions, chroma DC,
//!   MVD-suffix-LSB cascade-unsafe positions, or out-of-range
//!   positions. STC never picks these.
//!
//! The encoder feeds these costs into `combine_cover_4domain` for the
//! Scheme A combined STC, which then concentrates flips in the
//! highest-detectability-headroom regions.
//!
//! ## What this does NOT cover (yet)
//!
//! - **Motion compensation for P/B frames**: the wavelet cost uses
//!   the *input* YUV as the cover image (encoder-side). For
//!   reference-frame stego positions that depend on motion-compensated
//!   prediction, the proper input would be the *reconstructed* frame
//!   from the decoder. Approximation: use the input YUV everywhere.
//!   For typical short-message stego at typical bit-rates, this is a
//!   good enough proxy — the encoder's MC residual is close to the
//!   source frame for low-detail regions and divergent for textured
//!   regions, which is where we want low cost anyway. Refinement
//!   deferred to v1.1+.
//!
//! - **MVD-domain wavelet cost**: MVD positions don't have a pixel-
//!   block interpretation; instead we use the MB's aggregate Y-plane
//!   wavelet response as the cost proxy (lower texture = higher cost).
//!   This is a simpler heuristic than the retired CAVLC `h264_mvd_cost`
//!   module's residual-energy-based formula, but adequate for first-cut.
//!   Refinement deferred to v1.1+.
//!
//! ## Determinism
//!
//! All floating-point operations route through the same f64 paths +
//! `POW2_QBITS_MINUS_4` LUT that the existing
//! [`crate::stego::cost::h264_uniward`] uses. Cross-platform
//! deterministic by the same discipline as image J-UNIWARD.
//!
//! ## Memory
//!
//! Per-frame wavelet decomposition is ~32 MB at 1080p. Module
//! processes one frame at a time and drops the wavelet buffers before
//! moving to the next, keeping peak memory at ~32 MB regardless of
//! `n_frames`. The wavelet is parallelized *internally* (rows/cols in
//! [`crate::stego::cost::h264_uniward::compute_three_subbands`]) rather
//! than across frames, so this ~32 MB peak is preserved while still
//! using all cores (the convolution's per-row parallelism saturates
//! them on its own — frame-level concurrency would only add
//! `cores × 32 MB` for no throughput gain).

use crate::codec::h264::tables::BLOCK_INDEX_TO_POS;
use crate::codec::h264::stego::hook::{BinKind, EmbedDomain, PositionKey, SyntaxPath};
use crate::codec::h264::stego::inject::DomainCover;
use crate::codec::h264::stego::orchestrate::DomainCosts;
use crate::stego::cost::h264_uniward::{
    accumulate_cost_from_field, compute_delta_field, compute_three_subbands,
    precompute_unit_basis, DeltaField, ThreeSubbands,
};
use crate::stego::error::StegoError;
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Default `QP` value used when the caller doesn't pass one. Matches
/// `EncodeOpts::default().qp` (=26) in the OH264 path.
const DEFAULT_QP: i32 = 26;

/// Stabilisation constant for the MVD-domain texture proxy. Avoids
/// division by zero in flat-MB regions.
const MVD_COST_SIGMA: f32 = 1e-3;

/// Compute Tier 3 content-adaptive costs for the 4-domain cover.
///
/// `yuv` must be `width * height * 3 / 2 * n_frames` bytes in I420
/// (YUV420p) layout, with all `n_frames` frames concatenated. `width`
/// and `height` must be 16-aligned.
///
/// `cover` carries the 4-domain positions emitted by the walker. The
/// returned [`DomainCosts`] has the same per-domain lengths as
/// `cover.{coeff_sign_bypass, coeff_suffix_lsb, mvd_sign_bypass,
/// mvd_suffix_lsb}.bits.len()` — `costs[i]` is the cost for
/// `positions[i]`.
///
/// `qp` is the luma quantization parameter used to scale per-position
/// pixel-domain deltas. Pass the encoder's QP; falls back to 26 when
/// you don't know it.
///
/// # Errors
///
/// - [`StegoError::InvalidVideo`] if YUV length doesn't match
///   `width × height × 3/2 × n_frames` or dims are not 16-aligned.
pub fn compute_content_costs_yuv(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    cover: &DomainCover,
    qp: i32,
) -> Result<DomainCosts, StegoError> {
    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    if yuv.len() != frame_size * n_frames as usize {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {}", yuv.len(), frame_size * n_frames as usize,
        )));
    }

    let mb_width = (width / 16) as usize;
    let y_plane_size = (width * height) as usize;
    let chroma_plane_size = y_plane_size / 4;

    let unit_basis = precompute_unit_basis();
    // Memoise the position-independent J-UNIWARD impact field. It depends only
    // on (scan_pos, delta_class, qp); qp is constant for the whole call, so the
    // ≤15×33 distinct fields are built ONCE here (~4 MB, sub-ms) and looked up
    // per position instead of recomputing the 19×19 separable wavelet filter
    // for every coefficient position (millions on a 1080p clip). Bit-identical
    // to `compute_position_cost` — same arithmetic, just cached. See
    // [`coeff_position_cost`]. The table is shared read-only across frames and
    // the parallel position loop.
    let field_table = build_field_table(&unit_basis, qp);

    // Group positions by frame_idx, recording (domain, slot_in_domain).
    // We compute wavelets once per frame; positions visiting the same
    // frame look up costs into the cached wavelets.
    let n_csb = cover.coeff_sign_bypass.bits.len();
    let n_csl = cover.coeff_suffix_lsb.bits.len();
    let n_msb = cover.mvd_sign_bypass.bits.len();
    let n_msl = cover.mvd_suffix_lsb.bits.len();

    let mut costs = DomainCosts {
        coeff_sign_bypass: vec![f32::INFINITY; n_csb],
        coeff_suffix_lsb: vec![f32::INFINITY; n_csl],
        mvd_sign_bypass: vec![f32::INFINITY; n_msb],
        mvd_suffix_lsb: vec![f32::INFINITY; n_msl],
    };

    // #847 wavelet-share probe: PHASM_PERF_TRACE accumulates the time spent in
    // the (already row-parallel) 3-subband decomposition vs. the rest of
    // content_costs, to confirm whether the wavelet is the dominant cost.
    let trace_cc = std::env::var("PHASM_PERF_TRACE").as_deref() == Ok("1");
    let mut t_wav = std::time::Duration::ZERO;

    let frame_buckets = group_positions_by_frame(cover);
    for (frame_idx, bucket) in &frame_buckets {
        if *frame_idx as usize >= n_frames as usize {
            continue;
        }
        let yuv_offset = (*frame_idx as usize) * frame_size;
        let y_plane = &yuv[yuv_offset .. yuv_offset + y_plane_size];
        let cb_offset = yuv_offset + y_plane_size;
        let cb_plane = &yuv[cb_offset .. cb_offset + chroma_plane_size];
        let cr_offset = cb_offset + chroma_plane_size;
        let cr_plane = &yuv[cr_offset .. cr_offset + chroma_plane_size];

        // WV.6.b.3: the per-frame wavelet + cost work is shared with the spill
        // variant via `fill_frame_costs` (bit-identical — same arithmetic).
        let dt = fill_frame_costs(
            y_plane, cb_plane, cr_plane, bucket, cover, &field_table,
            width as usize, height as usize, mb_width, &mut costs,
        );
        if trace_cc {
            t_wav += dt;
        }
    }

    if trace_cc {
        eprintln!(
            "[PHASM_PERF_TRACE]     ↳ of which wavelet (3-subband decomp) {:>8.1} ms",
            t_wav.as_secs_f64() * 1000.0
        );
    }
    Ok(costs)
}

/// Per-frame wavelet decomposition + per-position cost evaluation for
/// [`compute_content_costs_yuv`] (one frame's planes in, this frame's positions
/// written into `costs` at their disjoint domain indices). Returns the time
/// spent in the 3-subband decomposition (for PHASM_PERF_TRACE). Factored out so
/// the WV.6.g per-GOP re-decode path can reuse the exact per-frame arithmetic.
#[allow(clippy::too_many_arguments)]
fn fill_frame_costs(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    bucket: &FrameBucket,
    cover: &DomainCover,
    field_table: &[DeltaField],
    width: usize,
    height: usize,
    mb_width: usize,
    costs: &mut DomainCosts,
) -> std::time::Duration {
    let tw = std::time::Instant::now();
    let y_wavelets = compute_three_subbands(y_plane, width, height);
    let cb_wavelets = compute_three_subbands(cb_plane, width / 2, height / 2);
    let cr_wavelets = compute_three_subbands(cr_plane, width / 2, height / 2);
    let wav_elapsed = tw.elapsed();

    // Coefficient-domain positions: per-position wavelet cost.
    // #perf 2026-06: this per-position loop is the cover-scaling serial
    // bottleneck (~82% of content_costs on high-texture footage). Each
    // position reads the frame's SHARED, read-only wavelets and writes a
    // DISJOINT cost index, so parallelising over positions is
    // bit-identical to the serial loop (per-position-deterministic
    // values; each index written exactly once) AND memory-neutral: the
    // wavelets are shared by reference, not duplicated, preserving the
    // module's one-frame-at-a-time ~32 MB peak. (Cross-frame concurrency
    // was rejected — see the module ## Memory note — because the wavelet
    // already parallelises internally; this parallelises the loop the
    // wavelet doesn't.)
    let cost_of = |slot: &CoverSlot| -> Option<(EmbedDomain, usize, f32)> {
        let CoverSlot { domain, idx } = *slot;
        let (positions, magnitudes) = match domain {
            EmbedDomain::CoeffSignBypass => (
                &cover.coeff_sign_bypass.positions,
                &cover.coeff_sign_bypass.magnitudes,
            ),
            EmbedDomain::CoeffSuffixLsb => (
                &cover.coeff_suffix_lsb.positions,
                &cover.coeff_suffix_lsb.magnitudes,
            ),
            _ => return None,
        };
        let &key = positions.get(idx)?;
        // `δ = 2·|coeff|` for Sign and `δ = 1` for SuffixLsb (the LSB
        // flip changes magnitude by ±1). Magnitude 0 only for legacy
        // `push(bit, pos)` call sites, which take the `δ = 2` fallback,
        // so cost-function behaviour is unchanged.
        let magnitude = magnitudes.get(idx).copied().unwrap_or(0);
        let cost = coeff_position_cost(
            key, magnitude, field_table,
            &y_wavelets, &cb_wavelets, &cr_wavelets,
            width, height, mb_width,
        );
        Some((domain, idx, cost))
    };
    #[cfg(feature = "parallel")]
    let coeff_contribs: Vec<(EmbedDomain, usize, f32)> =
        bucket.coeff_positions.par_iter().filter_map(&cost_of).collect();
    #[cfg(not(feature = "parallel"))]
    let coeff_contribs: Vec<(EmbedDomain, usize, f32)> =
        bucket.coeff_positions.iter().filter_map(&cost_of).collect();
    for &(domain, idx, cost) in &coeff_contribs {
        match domain {
            EmbedDomain::CoeffSignBypass => costs.coeff_sign_bypass[idx] = cost,
            EmbedDomain::CoeffSuffixLsb => costs.coeff_suffix_lsb[idx] = cost,
            _ => {}
        }
    }

    // MVD-domain positions: MB-aggregated wavelet cost. Parallelised
    // over positions like the coeff loop above (#perf 2026-06): each
    // position computes its MB texture directly from the frame's shared
    // read-only wavelets and writes a DISJOINT index — bit-identical and
    // memory-neutral. The previous per-MB texture cache is dropped:
    // `mb_y_texture` is cheap, recomputing it for the few MVD positions
    // that share an MB costs far less than the parallelism gain, and a
    // shared cache would re-serialise the loop.
    let mvd_cost_of = |slot: &CoverSlot| -> Option<(EmbedDomain, usize, f32)> {
        let CoverSlot { domain, idx } = *slot;
        let positions = match domain {
            EmbedDomain::MvdSignBypass => &cover.mvd_sign_bypass.positions,
            EmbedDomain::MvdSuffixLsb => &cover.mvd_suffix_lsb.positions,
            _ => return None,
        };
        let &key = positions.get(idx)?;
        let mb_addr = key.mb_addr();
        let mb_x = (mb_addr as usize) % mb_width;
        let mb_y = (mb_addr as usize) / mb_width;
        let texture = mb_y_texture(&y_wavelets, mb_x, mb_y, width, height);
        let cost = 1.0 / (texture + MVD_COST_SIGMA);
        Some((domain, idx, cost))
    };
    #[cfg(feature = "parallel")]
    let mvd_contribs: Vec<(EmbedDomain, usize, f32)> =
        bucket.mvd_positions.par_iter().filter_map(&mvd_cost_of).collect();
    #[cfg(not(feature = "parallel"))]
    let mvd_contribs: Vec<(EmbedDomain, usize, f32)> =
        bucket.mvd_positions.iter().filter_map(&mvd_cost_of).collect();
    for &(domain, idx, cost) in &mvd_contribs {
        match domain {
            EmbedDomain::MvdSignBypass => costs.mvd_sign_bypass[idx] = cost,
            EmbedDomain::MvdSuffixLsb => costs.mvd_suffix_lsb[idx] = cost,
            _ => {}
        }
    }

    wav_elapsed
}

/// Per-frame work bucket: indices into the original `DomainCover` for
/// each domain, partitioned by `frame_idx`. Keeps the cost-lookup loop
/// tight: one wavelet decomp per frame, all that frame's positions
/// resolved against it.
#[derive(Default)]
struct FrameBucket {
    /// (domain, slot index in DomainBits) for coefficient-domain
    /// positions (CSB + CSL).
    coeff_positions: Vec<CoverSlot>,
    /// (domain, slot index in DomainBits) for MVD-domain positions
    /// (MSB + MSL).
    mvd_positions: Vec<CoverSlot>,
}

#[derive(Copy, Clone)]
struct CoverSlot {
    domain: EmbedDomain,
    idx: usize,
}

fn group_positions_by_frame(cover: &DomainCover) -> HashMap<u32, FrameBucket> {
    let mut out: HashMap<u32, FrameBucket> = HashMap::new();
    for (i, &k) in cover.coeff_sign_bypass.positions.iter().enumerate() {
        out.entry(k.frame_idx()).or_default().coeff_positions
            .push(CoverSlot { domain: EmbedDomain::CoeffSignBypass, idx: i });
    }
    for (i, &k) in cover.coeff_suffix_lsb.positions.iter().enumerate() {
        out.entry(k.frame_idx()).or_default().coeff_positions
            .push(CoverSlot { domain: EmbedDomain::CoeffSuffixLsb, idx: i });
    }
    for (i, &k) in cover.mvd_sign_bypass.positions.iter().enumerate() {
        out.entry(k.frame_idx()).or_default().mvd_positions
            .push(CoverSlot { domain: EmbedDomain::MvdSignBypass, idx: i });
    }
    for (i, &k) in cover.mvd_suffix_lsb.positions.iter().enumerate() {
        out.entry(k.frame_idx()).or_default().mvd_positions
            .push(CoverSlot { domain: EmbedDomain::MvdSuffixLsb, idx: i });
    }
    out
}

/// Compute J-UNIWARD-style wavelet cost for a coefficient-domain
/// position. Returns `f32::INFINITY` for DC positions, chroma DC, and
/// positions whose syntax_path isn't a coefficient form.
fn coeff_position_cost(
    key: PositionKey,
    magnitude: u16,
    field_table: &[DeltaField],
    y_wavelets: &ThreeSubbands,
    cb_wavelets: &ThreeSubbands,
    cr_wavelets: &ThreeSubbands,
    img_w: usize,
    img_h: usize,
    mb_width: usize,
) -> f32 {
    let mb_addr = key.mb_addr();
    let mb_x = (mb_addr as usize) % mb_width;
    let mb_y = (mb_addr as usize) / mb_width;

    // Magnitude class for the field-table lookup. A sign flip swings the
    // coefficient +M↔−M (δ = 2·|coeff|); clamp |coeff| to 32 so a single
    // high-magnitude outlier can't dominate the cost vector (at QP ≤ 32,
    // coefficients above 32 are extremely rare; < 0.01% effect on STC plan
    // selection). Magnitude 0 (back-compat `push(bit, pos)` call sites)
    // clamps to 1 → δ=2, the legacy default. A SuffixLsb flip toggles the
    // magnitude LSB (δ=1) → class 0.
    let sign_class = (magnitude as usize).clamp(1, 32);

    let (wavelets, plane_w, plane_h, block_px_x, block_px_y, scan_pos, class) =
        match key.syntax_path() {
            SyntaxPath::Luma4x4 { block_idx, coeff_idx, kind } => {
                if coeff_idx == 0 {
                    // DC of luma 4x4 — kept wet.
                    return f32::INFINITY;
                }
                let (bx, by) = BLOCK_INDEX_TO_POS[block_idx as usize];
                let block_px_x = mb_x * 16 + bx as usize * 4;
                let block_px_y = mb_y * 16 + by as usize * 4;
                let class = match kind {
                    BinKind::Sign => sign_class,
                    BinKind::SuffixLsb => 0,
                };
                (y_wavelets, img_w, img_h, block_px_x, block_px_y, coeff_idx, class)
            }
            SyntaxPath::Luma8x8 { .. } => {
                // 8x8 transform — cost machinery currently targets 4x4
                // basis. Approximate by treating the 8x8 block as the
                // sub-4x4 at its top-left corner; covers the same
                // pixel region for cost-proxy purposes. v1.1 — add
                // proper 8x8 J-UNIWARD basis.
                return f32::INFINITY;
            }
            SyntaxPath::ChromaAc { plane, block_idx, coeff_idx, kind } => {
                if coeff_idx == 0 {
                    return f32::INFINITY;
                }
                let bx = (block_idx as usize) % 2;
                let by = (block_idx as usize) / 2;
                let chroma_w = img_w / 2;
                let chroma_h = img_h / 2;
                let block_px_x = mb_x * 8 + bx * 4;
                let block_px_y = mb_y * 8 + by * 4;
                let class = match kind {
                    BinKind::Sign => sign_class,
                    BinKind::SuffixLsb => 0,
                };
                let wv = if plane == 0 { cb_wavelets } else { cr_wavelets };
                (wv, chroma_w, chroma_h, block_px_x, block_px_y, coeff_idx, class)
            }
            SyntaxPath::ChromaDc { .. } => return f32::INFINITY,
            SyntaxPath::LumaDcIntra16x16 { .. } => return f32::INFINITY,
            SyntaxPath::Mvd { .. } => return f32::INFINITY,
        };

    // Look up the precomputed position-independent impact field for this
    // (scan_pos, class) and accumulate it against the cover wavelets at this
    // block — bit-identical to `compute_position_cost(unit_basis, …, δ)`.
    let field = &field_table[(scan_pos as usize - 1) * N_DELTA_CLASS + class];
    let cost =
        accumulate_cost_from_field(field, wavelets, plane_w, plane_h, block_px_x, block_px_y);
    if cost.is_finite() && cost > 0.0 { cost as f32 } else { f32::INFINITY }
}

/// Magnitude classes in the field table: class 0 = SuffixLsb (δ=1), classes
/// 1..=32 = Sign with clamped `|coeff|` (δ=2·class).
const N_DELTA_CLASS: usize = 33;

/// δ value for a field-table magnitude class. Mirrors `coeff_position_cost`'s
/// δ: class 0 → SuffixLsb (1.0); class c≥1 → Sign with clamped `|coeff|`=c (2·c).
#[inline]
fn delta_for_class(class: usize) -> f64 {
    if class == 0 { 1.0 } else { 2.0 * class as f64 }
}

/// Precompute every distinct [`DeltaField`] for the call: 15 AC scan positions
/// (1..=15; DC is kept wet) × 33 magnitude classes. `qp` is constant for the
/// call, so this is built once and shared read-only across all frames and the
/// parallel position loop. Lookup index: `(scan_pos - 1) * N_DELTA_CLASS + class`.
fn build_field_table(unit_basis: &[[[[f64; 4]; 4]; 4]; 4], qp: i32) -> Vec<DeltaField> {
    let mut table = Vec::with_capacity(15 * N_DELTA_CLASS);
    for scan_pos in 1u8..=15 {
        for class in 0..N_DELTA_CLASS {
            table.push(compute_delta_field(unit_basis, scan_pos, qp, delta_for_class(class)));
        }
    }
    table
}

/// MB-level aggregate texture from the Y-plane wavelet subbands.
/// Sums `|LH|+|HL|+|HH|` over the 16×16 pixel area of the MB. Larger
/// value = more texture = lower MVD-flip cost.
fn mb_y_texture(
    y_wavelets: &ThreeSubbands,
    mb_x: usize,
    mb_y: usize,
    img_w: usize,
    img_h: usize,
) -> f32 {
    let px0 = mb_x * 16;
    let py0 = mb_y * 16;
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for dy in 0..16 {
        let abs_y = py0 as isize + dy as isize;
        if abs_y < 0 || abs_y >= img_h as isize { continue; }
        let wy = (abs_y - y_wavelets.y_offset) as usize;
        for dx in 0..16 {
            let abs_x = px0 as isize + dx as isize;
            if abs_x < 0 || abs_x >= img_w as isize { continue; }
            let wx = (abs_x - y_wavelets.x_offset) as usize;
            let idx = wy * y_wavelets.width + wx;
            sum += y_wavelets.lh[idx].abs()
                 + y_wavelets.hl[idx].abs()
                 + y_wavelets.hh[idx].abs();
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_yuv(width: u32, height: u32, n_frames: u32) -> Vec<u8> {
        let frame_size = (width * height * 3 / 2) as usize;
        let mut out = Vec::with_capacity(frame_size * n_frames as usize);
        let w = width as i32;
        let h = height as i32;
        let half_w = w / 2;
        let half_h = h / 2;
        for f in 0..n_frames {
            for j in 0..h {
                for i in 0..w {
                    let v = ((i + f as i32 * 2) ^ (j + f as i32 * 3)) as u8;
                    out.push(v);
                }
            }
            let mut s: u32 = 0xCAFE_F00D ^ f;
            for _plane in 0..2 {
                for j in 0..half_h {
                    for i in 0..half_w {
                        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                        let tex = (s >> 16) as u8;
                        let pos = (i + j + f as i32) as u8;
                        out.push(tex.wrapping_add(pos));
                    }
                }
            }
        }
        out
    }

    fn empty_cover() -> DomainCover {
        DomainCover::default()
    }

    #[test]
    fn empty_cover_returns_empty_costs() {
        let yuv = synth_yuv(64, 64, 1);
        let cover = empty_cover();
        let costs = compute_content_costs_yuv(&yuv, 64, 64, 1, &cover, 26).unwrap();
        assert!(costs.coeff_sign_bypass.is_empty());
        assert!(costs.coeff_suffix_lsb.is_empty());
        assert!(costs.mvd_sign_bypass.is_empty());
        assert!(costs.mvd_suffix_lsb.is_empty());
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let yuv = vec![0u8; 100];
        let cover = empty_cover();
        let res = compute_content_costs_yuv(&yuv, 128, 80, 1, &cover, 26);
        assert!(matches!(res, Err(StegoError::InvalidVideo(_))));
    }

    #[test]
    fn non_aligned_dims_rejected() {
        let yuv = vec![0u8; 100];
        let cover = empty_cover();
        let res = compute_content_costs_yuv(&yuv, 17, 16, 1, &cover, 26);
        assert!(matches!(res, Err(StegoError::InvalidVideo(_))));
    }

    /// Construct a single Luma4x4 CSB position at (mb=0, block=0,
    /// coeff=1) and assert the cost is finite + positive on the
    /// synthetic-texture frame.
    #[test]
    fn single_luma4x4_csb_position_has_finite_cost() {
        let yuv = synth_yuv(64, 64, 1);
        let key = PositionKey::new(
            0, 0, EmbedDomain::CoeffSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 1, kind: BinKind::Sign },
        );
        let cover = DomainCover {
            coeff_sign_bypass: crate::codec::h264::stego::inject::DomainBits {
                bits: vec![0],
                positions: vec![key],
                magnitudes: vec![0],
            },
            ..Default::default()
        };
        let costs = compute_content_costs_yuv(&yuv, 64, 64, 1, &cover, 26).unwrap();
        assert_eq!(costs.coeff_sign_bypass.len(), 1);
        let c = costs.coeff_sign_bypass[0];
        assert!(c.is_finite() && c > 0.0, "expected finite + positive cost, got {c}");
    }

    /// DC positions (coeff_idx=0) always get WET (infinity). STC must
    /// never flip them.
    #[test]
    fn dc_positions_are_wet() {
        let yuv = synth_yuv(64, 64, 1);
        let key = PositionKey::new(
            0, 0, EmbedDomain::CoeffSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 0, kind: BinKind::Sign },
        );
        let cover = DomainCover {
            coeff_sign_bypass: crate::codec::h264::stego::inject::DomainBits {
                bits: vec![0],
                positions: vec![key],
                magnitudes: vec![0],
            },
            ..Default::default()
        };
        let costs = compute_content_costs_yuv(&yuv, 64, 64, 1, &cover, 26).unwrap();
        assert!(costs.coeff_sign_bypass[0].is_infinite());
    }

    /// Smoke: an MVD position gets a finite cost from the MB texture
    /// proxy.
    #[test]
    fn mvd_position_has_finite_cost() {
        let yuv = synth_yuv(64, 64, 1);
        let key = PositionKey::new(
            0, 0, EmbedDomain::MvdSignBypass,
            SyntaxPath::Mvd {
                list: 0,
                partition: 0,
                axis: crate::codec::h264::stego::hook::Axis::X,
                kind: BinKind::Sign,
            },
        );
        let cover = DomainCover {
            mvd_sign_bypass: crate::codec::h264::stego::inject::DomainBits {
                bits: vec![0],
                positions: vec![key],
                magnitudes: vec![0],
            },
            ..Default::default()
        };
        let costs = compute_content_costs_yuv(&yuv, 64, 64, 1, &cover, 26).unwrap();
        assert!(costs.mvd_sign_bypass[0].is_finite());
        assert!(costs.mvd_sign_bypass[0] > 0.0);
    }

    /// Determinism gate: two independent computations on identical
    /// input produce bit-identical output. The wavelet + cost machine
    /// must be reproducible on the same machine, and (by the
    /// `POW2_QBITS_MINUS_4` + f64-only discipline that
    /// `h264_uniward` follows) across machines.
    #[test]
    fn determinism_same_machine() {
        let yuv = synth_yuv(64, 64, 2);
        let mut positions = Vec::new();
        for f in 0..2 {
            for block in 0..16 {
                positions.push(PositionKey::new(
                    f, 0, EmbedDomain::CoeffSignBypass,
                    SyntaxPath::Luma4x4 {
                        block_idx: block, coeff_idx: 1, kind: BinKind::Sign,
                    },
                ));
            }
        }
        let cover = DomainCover {
            coeff_sign_bypass: crate::codec::h264::stego::inject::DomainBits {
                bits: vec![0; positions.len()],
                magnitudes: vec![0; positions.len()],
                positions,
            },
            ..Default::default()
        };
        let costs_a = compute_content_costs_yuv(&yuv, 64, 64, 2, &cover, 26).unwrap();
        let costs_b = compute_content_costs_yuv(&yuv, 64, 64, 2, &cover, 26).unwrap();
        assert_eq!(costs_a.coeff_sign_bypass, costs_b.coeff_sign_bypass);
    }

    /// Textured region should have LOWER cost than smooth region.
    /// Build a half-textured / half-smooth YUV, query a position in
    /// each region, compare costs.
    #[test]
    fn textured_region_has_lower_cost_than_smooth() {
        let width = 64u32;
        let height = 64u32;
        // Smooth top half (constant 128), textured bottom half (random).
        let mut yuv = vec![128u8; (width * height) as usize];
        let mut s: u32 = 0x12345678;
        for j in (height / 2)..height {
            for i in 0..width {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                yuv[(j * width + i) as usize] = (s >> 16) as u8;
            }
        }
        // Chroma planes: constant 128.
        yuv.resize((width * height * 3 / 2) as usize, 128);

        // Smooth-region position: MB (0, 0).
        let smooth_key = PositionKey::new(
            0, 0, EmbedDomain::CoeffSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 1, kind: BinKind::Sign },
        );
        // Textured-region position: MB (0, 2) — y starts at row 32
        // (in 16-pixel MB units that's mb_y=2).
        let mb_width = width / 16;
        let textured_mb_addr = 2 * mb_width;
        let textured_key = PositionKey::new(
            0, textured_mb_addr, EmbedDomain::CoeffSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 1, kind: BinKind::Sign },
        );

        let cover = DomainCover {
            coeff_sign_bypass: crate::codec::h264::stego::inject::DomainBits {
                bits: vec![0, 0],
                positions: vec![smooth_key, textured_key],
                magnitudes: vec![0, 0],
            },
            ..Default::default()
        };
        let costs = compute_content_costs_yuv(&yuv, width, height, 1, &cover, 26).unwrap();
        let smooth_cost = costs.coeff_sign_bypass[0];
        let textured_cost = costs.coeff_sign_bypass[1];
        assert!(smooth_cost.is_finite());
        assert!(textured_cost.is_finite());
        assert!(
            textured_cost < smooth_cost,
            "textured cost {textured_cost} should be < smooth cost {smooth_cost}",
        );
    }
}
