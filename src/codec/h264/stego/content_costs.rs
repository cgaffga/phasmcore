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
use crate::codec::h264::stego::inject::{DomainBits, DomainCover};
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

/// Stabilisation constant for the MVD-domain texture proxy. Avoids a
/// zero-cost position in perfectly flat macroblocks, which STC would
/// otherwise flood.
const MVD_COST_SIGMA: f32 = 1e-3;

/// Scale bringing the MVD damage model onto the same footing as the
/// coefficient costs so the domain weights mean what they say.
///
/// Measured on `iphone7_smoke` (`core/tests/mvd_cost_model.rs`, 2026-10-20),
/// per-position cost as the STC trellis receives it, BEFORE domain weights:
///
/// ```text
///   CoeffSign   median 2149.8   (range 90.9 .. 28022.8)
///   MvdSign     median    0.1485 (range  0.019 ..     1.75)
/// ```
///
/// After the shipped 1:3:10:10 domain weights that is still ~1400x cheaper,
/// so STC spent its budget on MVD positions FIRST — the exact opposite of
/// what `cost_weights.rs` documents ("MvdSign / MvdSuffix only fire when
/// CoeffSign capacity is exhausted"). A 10x domain weight cannot bridge a
/// 14000x scale gap. This constant is what makes the weights the actual
/// knob again; it is a normalisation, not a tuning parameter.
const MVD_COST_SCALE: f32 = 1.0;

/// How far a flip in each MVD domain moves the macroblock's prediction,
/// in quarter-pel units.
///
/// * `MvdSignBypass` sends the vector from `+d` to `-d`, so the block
///   displaces by `2·|d|`. On the smoke fixture `|d|` reaches 55 — a
///   **27-pixel** jump, and under the old cost function those were among
///   the CHEAPEST positions in the whole cover.
/// * `MvdSuffixLsb` flips the low bit of the Exp-Golomb suffix, moving the
///   block by a single quarter-pel no matter how large the vector is.
///
/// That asymmetry is enormous and the shipped model did not represent it at
/// all: both domains carried the same weight and the same `1/texture` cost.
fn mvd_displacement_qpel(domain: EmbedDomain, magnitude: u16) -> f32 {
    match domain {
        EmbedDomain::MvdSignBypass => 2.0 * magnitude as f32,
        // Suffix LSB: ±1 quarter-pel, independent of |d|.
        _ => 1.0,
    }
}

/// Predicted squared reconstruction error of an MVD flip.
///
/// The wire-only emit changes the motion vector WITHOUT recomputing the
/// residual (by design — the encoder's state stays clean). So the decoder
/// reconstructs `pred(MV') + residual(MV)` and the error is
///
/// ```text
///     err = pred(MV) - pred(MV')  ≈  |∇I| · Δ
/// ```
///
/// the local spatial gradient times the displacement. `texture` (mean
/// |LH|+|HL|+|HH| over the macroblock) is a gradient-magnitude proxy, so
/// squared error goes as `(texture · Δpx)²`.
///
/// This is the opposite relationship to the one the code shipped with.
/// `1/(texture + σ)` is the right instinct for a COEFFICIENT change, which
/// perturbs one block's own texture and is masked by busy content. A motion
/// vector is not masked by busy content — busy content is precisely what
/// makes a displacement visible, because the prediction it lands on no
/// longer matches. Measured correlation between the old cost and this model
/// was NEGATIVE (-0.08 on MvdSign, -0.36 on MvdSuffix): the shipped function
/// preferred exactly the positions this one penalises.
fn mvd_position_cost(texture: f32, displacement_qpel: f32) -> f32 {
    let err = texture * (displacement_qpel / 4.0);
    (MVD_COST_SCALE * err * err).max(MVD_COST_SIGMA)
}

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
        let magnitudes = match domain {
            EmbedDomain::MvdSignBypass => &cover.mvd_sign_bypass.magnitudes,
            EmbedDomain::MvdSuffixLsb => &cover.mvd_suffix_lsb.magnitudes,
            _ => return None,
        };
        let &key = positions.get(idx)?;
        let mb_addr = key.mb_addr();
        let mb_x = (mb_addr as usize) % mb_width;
        let mb_y = (mb_addr as usize) / mb_width;
        let texture = mb_y_texture(&y_wavelets, mb_x, mb_y, width, height);
        let magnitude = magnitudes.get(idx).copied().unwrap_or(0);
        let cost = mvd_position_cost(texture, mvd_displacement_qpel(domain, magnitude));
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

/// Structural ∞/finite mask over the 4-domain cover — the capacity probe's
/// stand-in for [`compute_content_costs_yuv`].
///
/// ## Why this is exact, not an approximation
///
/// The capacity probe ([`super::super::openh264_stego::h264_gop_capacity`])
/// consumes costs for exactly one purpose: deciding whether
/// `stc_embed(...).is_some()` for a trial payload, bisected to find the
/// largest embeddable size. `stc_embed`'s inline path fails on **one**
/// condition — `best_cost == inf`, i.e. no finite-cost path exists through
/// the Viterbi trellis (`crate::stego::stc::embed`). Finite cost
/// *magnitudes* steer which flips STC picks; they cannot change whether a
/// solution exists. `combine_bits_4domain` only multiplies each cost by a
/// positive per-domain weight, which maps finite→finite and ∞→∞.
///
/// So the probe's answer is a function of the cost vector's ∞ pattern alone,
/// and that pattern is structural: it follows from each position's
/// [`SyntaxPath`] (DC coefficients, 8×8 luma, chroma DC, luma DC of
/// Intra16×16 and MVD paths are kept wet) plus the out-of-range frame check —
/// none of which needs the YUV planes, the wavelet decomposition or the
/// delta-field accumulation. Skipping them makes the probe's cost step O(n)
/// pointer-chasing instead of a per-GOP wavelet + 19×19×3 per-position
/// reduction, which measured at **41% of total probe wall time** on 1080p
/// (2026-10-18: 15.8 s of 38.5 s, `PHASM_CAP_PROFILE=1`).
///
/// MVD-domain positions are always finite here: `fill_frame_costs` gives them
/// `mvd_position_cost(texture, displacement)`, which is a product of finite
/// non-negative terms floored at `MVD_COST_SIGMA` — finite and non-zero, and
/// positive for every in-range position. The cascade-safety `safe_msl` gate
/// that wets most of `mvd_suffix_lsb` is applied by the caller *after* this,
/// exactly as it is applied after the real costs.
///
/// The one place this could in principle diverge from the real computation is
/// [`coeff_position_cost`]'s data-dependent tail, which maps a computed cost
/// to ∞ unless it is finite and `> 0.0`. That cannot fire for real content (a
/// zero cost needs an identically-zero delta field over the whole in-bounds
/// 19×19 window), but "cannot" is not "does not": the equivalence is pinned
/// empirically over the real-world corpus by
/// `core/tests/cap2_mask_equivalence.rs`, which asserts identical
/// `per_tier_payload` from both paths. If that gate ever fires, the probe is
/// wrong — fix it, don't re-baseline it.
pub(crate) fn compute_cost_mask(cover: &DomainCover, n_frames: u32) -> DomainCosts {
    // `1.0`, not the real cost: any finite value yields the same reachability,
    // and a constant makes it obvious at a glance that magnitudes are unused.
    const FINITE: f32 = 1.0;

    /// Mirrors [`coeff_position_cost`]'s ∞ returns for the coefficient
    /// domains. Kept adjacent to it on purpose — the two must agree, and a
    /// new wet `SyntaxPath` arm has to be added in both places.
    ///
    /// ## Why each arm is wet — all three reasons are INTENTIONAL
    ///
    /// Established 2026-10-19 by code reading + a per-domain wet census over
    /// the real-world corpus at native 1080p (horse_flag / iphone7 / lumix_g9,
    /// 30f, QP 26). Recording it here because the rationale differs per arm
    /// and the predicate had no note at this call site.
    ///
    /// 1. **DC positions** — `Luma4x4 { coeff_idx: 0 }`,
    ///    `ChromaAc { coeff_idx: 0 }`, `ChromaDc`, `LumaDcIntra16x16`. Two
    ///    independent reasons, either sufficient: (a) the J-UNIWARD field
    ///    table ([`build_field_table`]) is built for AC scan positions
    ///    `1..=15` only — there is no DC entry to look up, by construction;
    ///    (b) a DC sign flip propagates *coherently* over the whole parent
    ///    macroblock (256 luma / 64 chroma pixels shift brightness by
    ///    2N·qstep) instead of the mostly-cancelling 16-pixel IDCT response
    ///    of an AC flip — the `-8.77 dB` IDR damage that motivated the 16×
    ///    DC cost multiplier in `orchestrate::ResidualPathKind::cost_multiplier`
    ///    (commit a29c7748). **This, not `Luma8x8`, is what drives the
    ///    measured wet fraction**: of the 70.1% wet `CoeffSignBypass`
    ///    positions on horse_flag, 131354 are Luma4x4 DC, 124411
    ///    `LumaDcIntra16x16`, 47479 `ChromaDc` and 2528 `ChromaAc{0}` —
    ///    and **zero** are `Luma8x8`.
    ///
    /// 2. **`Luma8x8`** — a missing 8×8 J-UNIWARD cost model, NOT a
    ///    cascade-safety exclusion. The wavelet machinery
    ///    ([`crate::stego::cost::h264_uniward`]) is built on the 4×4 integer
    ///    transform basis; there is no 8×8 basis to compute a cost against,
    ///    so the position cannot be priced and is kept wet rather than
    ///    mispriced. Wet since the predicate's introduction (commit 817b3431,
    ///    STEGO.A.1) — see the `v1.1` note on
    ///    [`coeff_position_cost`]'s matching arm.
    ///
    ///    In the **production OH264 path this arm is vacuous**: the pinned
    ///    OpenH264 fork has no 8×8-transform implementation at all
    ///    (`Transform8x8`/`DCT8x8` do not exist in the encoder source), so it
    ///    emits `transform_8x8_mode_flag = 0` in every PPS and block
    ///    category 5 never occurs — see the encoder-contract note in
    ///    `core/openh264-sys/src/lib.rs` and
    ///    `openh264.rs`'s "none expected at our config". The census confirms
    ///    it empirically: 0 `Luma8x8` positions on all three fixtures. The
    ///    arm earns its keep only because the CABAC *walker* parses I_8x8
    ///    residual from arbitrary Annex-B input
    ///    (`cabac::bin_decoder::slice`, gated on the stream's own
    ///    `pps.transform_8x8_mode_flag`), so a third-party High-profile cover
    ///    can reach it. Dropping the arm would price those positions with the
    ///    wrong basis; adding an 8×8 basis is the only real fix.
    ///
    /// 3. **`Mvd`** — unreachable here (this predicate is only ever applied to
    ///    the two coefficient domains; MVD positions go through `mvd_mask`,
    ///    which keeps them finite). Kept as a defensive arm so the `match`
    ///    stays exhaustive over `SyntaxPath`.
    ///
    /// Capacity consequence, so a future reader does not "fix" it: these ∞
    /// positions are wet for the **primary STC only**. The shadow layer
    /// selects positions with no reference to this cost vector at all
    /// (`shadow::for_each_eligible_position` takes cover + cascade masks, no
    /// costs), so `GopCapacity::injectable_cover_bits` is right to count
    /// them — ~46-60% of the slots a real shadow encode uses are ∞-cost here.
    fn coeff_is_wet(key: PositionKey) -> bool {
        match key.syntax_path() {
            SyntaxPath::Luma4x4 { coeff_idx, .. } => coeff_idx == 0,
            SyntaxPath::ChromaAc { coeff_idx, .. } => coeff_idx == 0,
            SyntaxPath::Luma8x8 { .. }
            | SyntaxPath::ChromaDc { .. }
            | SyntaxPath::LumaDcIntra16x16 { .. }
            | SyntaxPath::Mvd { .. } => true,
        }
    }

    let coeff_mask = |positions: &[PositionKey]| -> Vec<f32> {
        positions
            .iter()
            .map(|&k| {
                if k.frame_idx() >= n_frames || coeff_is_wet(k) {
                    f32::INFINITY
                } else {
                    FINITE
                }
            })
            .collect()
    };
    let mvd_mask = |positions: &[PositionKey]| -> Vec<f32> {
        positions
            .iter()
            .map(|&k| if k.frame_idx() >= n_frames { f32::INFINITY } else { FINITE })
            .collect()
    };

    DomainCosts {
        coeff_sign_bypass: coeff_mask(&cover.coeff_sign_bypass.positions),
        coeff_suffix_lsb: coeff_mask(&cover.coeff_suffix_lsb.positions),
        mvd_sign_bypass: mvd_mask(&cover.mvd_sign_bypass.positions),
        mvd_suffix_lsb: mvd_mask(&cover.mvd_suffix_lsb.positions),
    }
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
                // 8×8 transform — the cost machinery targets the 4×4 basis,
                // so there is nothing correct to price this against: kept
                // WET (intentional). v1.1 — add a proper 8×8 J-UNIWARD
                // basis; until then do NOT substitute a 4×4 proxy (an
                // earlier draft of this comment proposed re-using the
                // top-left sub-4×4, which was never implemented and would
                // misprice the other 48 coefficients).
                //
                // Vacuous on the production OH264 cover — the pinned fork
                // emits `transform_8x8_mode_flag = 0` always, so BC=5 never
                // occurs. Reachable only via the walker on third-party
                // High-profile input. Full rationale + the corpus census:
                // `compute_cost_mask::coeff_is_wet`.
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
mod prefix_cache_tests {
    use super::*;

    use crate::stego::gop_cache::{BoundedGopCache, GopCachePayload};

    fn dom(n: usize, seed: u64) -> DomainBits {
        DomainBits {
            bits: (0..n).map(|i| ((i as u64 + seed) % 2) as u8).collect(),
            positions: (0..n)
                .map(|i| PositionKey::from_raw(0x0123_4567_89AB_CDEF ^ (i as u64) ^ seed))
                .collect(),
            magnitudes: (0..n).map(|i| ((i * 7 + seed as usize) % 65535) as u16).collect(),
        }
    }

    fn costs(n: usize, with_inf: bool) -> Vec<f32> {
        (0..n)
            .map(|i| {
                if with_inf && i % 3 == 0 {
                    f32::INFINITY
                } else {
                    // Deliberately awkward values: denormals and a negative
                    // zero would survive a to_bits round trip but not a
                    // decimal one.
                    (i as f32) * 1.000_000_1_f32 + f32::MIN_POSITIVE
                }
            })
            .collect()
    }

    fn sample() -> ShadowGopPrefix {
        ShadowGopPrefix {
            cover: DomainCover {
                coeff_sign_bypass: dom(37, 1),
                coeff_suffix_lsb: dom(5, 2),
                mvd_sign_bypass: dom(11, 3),
                mvd_suffix_lsb: dom(0, 4), // empty domain must round-trip too
            },
            costs: DomainCosts {
                coeff_sign_bypass: costs(37, true),
                coeff_suffix_lsb: costs(5, true),
                mvd_sign_bypass: costs(11, false),
                mvd_suffix_lsb: Vec::new(),
            },
        }
    }

    fn assert_same(a: &ShadowGopPrefix, b: &ShadowGopPrefix) {
        for (name, x, y) in [
            ("csb", &a.cover.coeff_sign_bypass, &b.cover.coeff_sign_bypass),
            ("csl", &a.cover.coeff_suffix_lsb, &b.cover.coeff_suffix_lsb),
            ("msb", &a.cover.mvd_sign_bypass, &b.cover.mvd_sign_bypass),
            ("msl", &a.cover.mvd_suffix_lsb, &b.cover.mvd_suffix_lsb),
        ] {
            assert_eq!(x.bits, y.bits, "{name} bits");
            assert_eq!(
                x.positions.iter().map(|p| p.raw()).collect::<Vec<_>>(),
                y.positions.iter().map(|p| p.raw()).collect::<Vec<_>>(),
                "{name} positions"
            );
            assert_eq!(x.magnitudes, y.magnitudes, "{name} magnitudes");
        }
        for (name, x, y) in [
            ("csb", &a.costs.coeff_sign_bypass, &b.costs.coeff_sign_bypass),
            ("csl", &a.costs.coeff_suffix_lsb, &b.costs.coeff_suffix_lsb),
            ("msb", &a.costs.mvd_sign_bypass, &b.costs.mvd_sign_bypass),
            ("msl", &a.costs.mvd_suffix_lsb, &b.costs.mvd_suffix_lsb),
        ] {
            assert_eq!(x.len(), y.len(), "{name} cost len");
            for (i, (p, q)) in x.iter().zip(y.iter()).enumerate() {
                // BIT equality, not ==: ∞ must survive exactly (it is what
                // marks a position unflippable, so a lossy round trip would
                // change which flips STC picks), and NaN != NaN under `==`.
                assert_eq!(p.to_bits(), q.to_bits(), "{name} cost[{i}] bits");
            }
        }
    }

    /// The payload must survive pack → unpack bit-for-bit. A wrong unpack
    /// would hand sweep B a plausible-but-different cover and emit a file
    /// that encodes without error and decodes to garbage.
    #[test]
    fn shadow_gop_prefix_round_trips_bit_exactly() {
        let a = sample();
        let bytes = a.flat_pack();
        let b = ShadowGopPrefix::flat_unpack(&bytes).expect("unpack");
        assert_same(&a, &b);
    }

    #[test]
    fn shadow_gop_prefix_rejects_truncation_and_garbage() {
        let bytes = sample().flat_pack();
        assert!(ShadowGopPrefix::flat_unpack(&[]).is_err(), "empty");
        assert!(ShadowGopPrefix::flat_unpack(&[99]).is_err(), "bad version");
        for cut in [1usize, 8, bytes.len() / 2, bytes.len() - 1] {
            assert!(
                ShadowGopPrefix::flat_unpack(&bytes[..cut]).is_err(),
                "truncated at {cut} must not unpack"
            );
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(ShadowGopPrefix::flat_unpack(&extra).is_err(), "trailing byte");
    }

    /// End-to-end through the cache, including the disk tier — this is the
    /// path that actually runs when the RAM budget is small.
    #[test]
    fn round_trips_through_the_disk_tier() {
        let scratch = std::env::var("PHASM_TMP")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"));
        let a = sample();
        // ram_budget = 0 forces the disk tier.
        let mut cache: BoundedGopCache<ShadowGopPrefix> =
            BoundedGopCache::new(2, 0, 1 << 20, scratch, "h264-prefix-test");
        cache.insert(0, a.clone());
        assert!(cache.spilled(), "should have used the disk tier");
        let b = cache.fetch(0).expect("disk hit");
        assert_same(&a, &b);
        assert_eq!(cache.stats().disk_hits, 1);
    }

    #[test]
    fn cached_size_matches_the_15_bytes_per_position_model() {
        let p = ShadowGopPrefix {
            cover: DomainCover {
                coeff_sign_bypass: dom(100, 0),
                ..Default::default()
            },
            costs: DomainCosts { coeff_sign_bypass: costs(100, false), ..Default::default() },
        };
        // 100 positions x (1 bit + 8 position + 2 magnitude) + 100 x 4 costs
        assert_eq!(p.cached_size(), 100 * 11 + 100 * 4);
    }
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

// ═════════════════════════════════════════════════════════════════════════
// Shadow-sweep GOP cache payload
// ═════════════════════════════════════════════════════════════════════════

/// The per-GOP prefix both shadow sweeps compute identically, cached so the
/// second sweep can skip it.
///
/// `gop_provisional_step` (sweep A) and `sweep_b_emit` (sweep B) each run the
/// same five steps per GOP: clean OpenH264 encode → CABAC walk →
/// [`compute_content_costs_yuv`] → `analyze_safe_mvd_subset` /
/// `derive_msl_safe_from_msb` → `apply_safe_msl_gate`. Measured on a 2-GOP
/// shadow encode, the wavelet inside step 3 alone was 3.83 s of 12.39 s wall —
/// exactly 2 GOPs × 2 sweeps.
///
/// **What is NOT in here, and why:**
/// * `clean_bytes` — consumed only to produce the walk; nothing downstream
///   reads it.
/// * `mvd_meta` / `mb_w` / `mb_h` — consumed only to derive the safe-MSL
///   baseline, which is already applied to `costs`.
/// * the **tier** gate — `apply_tier_gate` is the one genuinely per-tier step,
///   and sweep B runs per parity tier. So the cache holds costs **after** the
///   safe-MSL gate and **before** the tier gate, and the caller applies the
///   tier gate on every fetch. Caching post-tier would make each entry valid
///   for one tier only and multiply the footprint by the tier count.
///
/// Footprint: `DomainBits` is 11 B/position (`bits` u8 + `positions`
/// [`PositionKey`] u64 + `magnitudes` u16) and `DomainCosts` is 4 B/position,
/// so ~15 B/position — about **6.9 MB** for a 1080p 30-frame GOP (~461k
/// positions). See `docs/design/video/gop-cache-budgets.md`.
#[derive(Clone)]
pub struct ShadowGopPrefix {
    pub cover: DomainCover,
    /// Post-safe-MSL, pre-tier. See the type docs.
    pub costs: DomainCosts,
}

/// Flat little-endian layout. Version-prefixed so a stale file hard-fails
/// rather than unpacking into plausible garbage; there is no cross-run
/// compatibility obligation (the scratch tree is removed on drop), so the
/// version only has to change when this layout does.
const SHADOW_GOP_PREFIX_FORMAT_V1: u8 = 1;

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn take_u32(b: &[u8], at: &mut usize) -> Result<u32, String> {
    let end = at.checked_add(4).ok_or("u32 offset overflow")?;
    let s = b.get(*at..end).ok_or("truncated u32")?;
    *at = end;
    Ok(u32::from_le_bytes(s.try_into().map_err(|_| "u32 slice")?))
}

fn pack_domain(out: &mut Vec<u8>, d: &DomainBits) {
    // The three vectors are index-aligned by construction (`DomainBits::len`
    // debug-asserts it), but pack each length anyway: a mismatch must surface
    // as an unpack error, not as a silently truncated cover.
    put_u32(out, d.bits.len() as u32);
    put_u32(out, d.positions.len() as u32);
    put_u32(out, d.magnitudes.len() as u32);
    out.extend_from_slice(&d.bits);
    for p in &d.positions {
        out.extend_from_slice(&p.raw().to_le_bytes());
    }
    for m in &d.magnitudes {
        out.extend_from_slice(&m.to_le_bytes());
    }
}

fn unpack_domain(b: &[u8], at: &mut usize) -> Result<DomainBits, String> {
    let n_bits = take_u32(b, at)? as usize;
    let n_pos = take_u32(b, at)? as usize;
    let n_mag = take_u32(b, at)? as usize;

    let end = at.checked_add(n_bits).ok_or("bits overflow")?;
    let bits = b.get(*at..end).ok_or("truncated bits")?.to_vec();
    *at = end;

    let mut positions = Vec::with_capacity(n_pos);
    for _ in 0..n_pos {
        let end = at.checked_add(8).ok_or("position overflow")?;
        let s = b.get(*at..end).ok_or("truncated position")?;
        *at = end;
        positions.push(PositionKey::from_raw(u64::from_le_bytes(
            s.try_into().map_err(|_| "position slice")?,
        )));
    }

    let mut magnitudes = Vec::with_capacity(n_mag);
    for _ in 0..n_mag {
        let end = at.checked_add(2).ok_or("magnitude overflow")?;
        let s = b.get(*at..end).ok_or("truncated magnitude")?;
        *at = end;
        magnitudes.push(u16::from_le_bytes(s.try_into().map_err(|_| "magnitude slice")?));
    }

    Ok(DomainBits { bits, positions, magnitudes })
}

fn pack_costs(out: &mut Vec<u8>, c: &[f32]) {
    put_u32(out, c.len() as u32);
    for v in c {
        // to_bits, not to_le_bytes on the float: ∞ and the exact bit pattern
        // must survive, because ∞ is load-bearing (it is what marks a position
        // unflippable) and a lossy round trip would change which flips STC
        // picks.
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

fn unpack_costs(b: &[u8], at: &mut usize) -> Result<Vec<f32>, String> {
    let n = take_u32(b, at)? as usize;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let end = at.checked_add(4).ok_or("cost overflow")?;
        let s = b.get(*at..end).ok_or("truncated cost")?;
        *at = end;
        v.push(f32::from_bits(u32::from_le_bytes(
            s.try_into().map_err(|_| "cost slice")?,
        )));
    }
    Ok(v)
}

impl crate::stego::gop_cache::GopCachePayload for ShadowGopPrefix {
    fn cached_size(&self) -> usize {
        let dom = |d: &DomainBits| d.bits.len() + d.positions.len() * 8 + d.magnitudes.len() * 2;
        dom(&self.cover.coeff_sign_bypass)
            + dom(&self.cover.coeff_suffix_lsb)
            + dom(&self.cover.mvd_sign_bypass)
            + dom(&self.cover.mvd_suffix_lsb)
            + (self.costs.coeff_sign_bypass.len()
                + self.costs.coeff_suffix_lsb.len()
                + self.costs.mvd_sign_bypass.len()
                + self.costs.mvd_suffix_lsb.len())
                * 4
    }

    fn flat_pack(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.cached_size() + 64);
        out.push(SHADOW_GOP_PREFIX_FORMAT_V1);
        pack_domain(&mut out, &self.cover.coeff_sign_bypass);
        pack_domain(&mut out, &self.cover.coeff_suffix_lsb);
        pack_domain(&mut out, &self.cover.mvd_sign_bypass);
        pack_domain(&mut out, &self.cover.mvd_suffix_lsb);
        pack_costs(&mut out, &self.costs.coeff_sign_bypass);
        pack_costs(&mut out, &self.costs.coeff_suffix_lsb);
        pack_costs(&mut out, &self.costs.mvd_sign_bypass);
        pack_costs(&mut out, &self.costs.mvd_suffix_lsb);
        out
    }

    fn flat_unpack(bytes: &[u8]) -> Result<Self, String> {
        let ver = *bytes.first().ok_or("empty payload")?;
        if ver != SHADOW_GOP_PREFIX_FORMAT_V1 {
            return Err(format!("unknown ShadowGopPrefix format {ver}"));
        }
        let mut at = 1usize;
        let cover = DomainCover {
            coeff_sign_bypass: unpack_domain(bytes, &mut at)?,
            coeff_suffix_lsb: unpack_domain(bytes, &mut at)?,
            mvd_sign_bypass: unpack_domain(bytes, &mut at)?,
            mvd_suffix_lsb: unpack_domain(bytes, &mut at)?,
        };
        let costs = DomainCosts {
            coeff_sign_bypass: unpack_costs(bytes, &mut at)?,
            coeff_suffix_lsb: unpack_costs(bytes, &mut at)?,
            mvd_sign_bypass: unpack_costs(bytes, &mut at)?,
            mvd_suffix_lsb: unpack_costs(bytes, &mut at)?,
        };
        if at != bytes.len() {
            return Err(format!("trailing bytes: consumed {at} of {}", bytes.len()));
        }
        Ok(Self { cover, costs })
    }
}
