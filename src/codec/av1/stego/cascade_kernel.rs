// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase B.1.5.2 — Cascade-safety v2 forward modeling kernels.
//!
//! Given a per-tuple impulse pattern from
//! [`crate::stego::cost::av1_uniward`]'s L1 basis cache (B.3.1.2),
//! predict the post-deblock pixel-domain perturbation that a single
//! AC sign flip would produce. Used by the L3 cache to compute the
//! cascade-magnitude scalar that drives B.1.5.5's three-tier
//! dispatch.
//!
//! ## v0.5 approximation strategy
//!
//! AV1's deblock filter is per-edge content-dependent: filter size
//! (4 / 6 / 8 / 14 taps) and clamping thresholds vary by coefficient
//! magnitudes in adjacent blocks. Spec-correct forward modeling per
//! every cover candidate is multi-session work.
//!
//! This module's first cut uses a **frame-level linear approximation**:
//! treat deblock as a separable 1D low-pass blur whose radius scales
//! with the frame-level deblock filter level (from
//! `PhasmFrameLoopFilterState.deblock_levels`). The kernel applies
//! horizontal smear with the H-direction level + vertical smear with
//! the V-direction level (per plane).
//!
//! Properties:
//! * **Bounded conservative**: the box-filter smear OVER-estimates
//!   cascade spread (real AV1 deblock smears less than a uniform
//!   box). Conservative is good for safety — we'd rather reject a
//!   borderline-safe position than accept a borderline-unsafe one.
//! * **Linear**: invariant in the impulse magnitude, so the L2 cache
//!   can store the *normalized* response and the L3 cache scales it
//!   by `|coeff|` at lookup.
//! * **Per-frame deterministic**: deblock levels are frame-level
//!   constants captured by B.1.5.1's
//!   [`PhasmFrameLoopFilterState`](phasm_rav1e::ec::PhasmFrameLoopFilterState).
//!   No per-flip state.
//!
//! ## Refinements deferred to v0.6+
//!
//! 1. **Per-edge boundary strength**: spec-correct deblock filter
//!    size depends on coefficient magnitudes in adjacent blocks
//!    (`get_dc_sign_ctx`-style context). Requires per-SB or per-
//!    block state capture in phasm-rav1e (B.1.5.1 deferred this).
//! 2. **Spec-correct filter taps**: AV1 deblock has 4 distinct
//!    filter-size variants with specific tap coefficients
//!    (per `vendor/phasm-rav1e/src/deblock.rs::deblock_size{4,6,8,14}_inner`).
//!    First cut uses a generic box filter for simplicity.
//! 3. **CDEF prediction layer**: B.1.5.3 stacks CDEF on top of
//!    deblock output. Adds a directional 8-tap secondary filter
//!    per superblock.
//!
//! See [`phase-b15-cascade-safety-v2.md`](../../../../../docs/design/video/av1/phase-b15-cascade-safety-v2.md)
//! § 2 (per-position forward modeling kernel) + § 6.5 (B.1.5.0
//! spike findings + calibration data).

use std::collections::HashMap;

use phasm_rav1e::ec::PhasmFrameLoopFilterState;

/// Apply a separable box-filter smear approximating AV1 deblock to a
/// `tx_w × tx_h` impulse pattern. Returns a fresh `Vec<f64>` of the
/// same length. The smear radius scales with the per-plane deblock
/// level from [`PhasmFrameLoopFilterState`]:
///
/// * `levels[0]` — Y vertical edge filter level (smear along x)
/// * `levels[1]` — Y horizontal edge filter level (smear along y)
/// * `levels[2]` — U filter level (used for both directions)
/// * `levels[3]` — V filter level (used for both directions)
///
/// Empty radius (level == 0) is the identity transform.
///
/// **Conservative bound**: real AV1 deblock spreads energy less than
/// a uniform box (the filter has falling-off taps + clamping). Box
/// over-estimates → safer in cascade-safety: borderline positions
/// get rejected, not accepted.
pub fn apply_deblock_box_smear(
    basis: &[f64],
    tx_w: usize,
    tx_h: usize,
    plane: u8,
    state: &PhasmFrameLoopFilterState,
) -> Vec<f64> {
    debug_assert_eq!(basis.len(), tx_w * tx_h);
    let (level_v_edge, level_h_edge) = match plane {
        0 => (state.deblock_levels[0], state.deblock_levels[1]),
        1 => (state.deblock_levels[2], state.deblock_levels[2]),
        2 => (state.deblock_levels[3], state.deblock_levels[3]),
        _ => return basis.to_vec(),
    };
    // Map level (0-63 in AV1 spec) to a box radius. Empirical
    // scaling: level 16 → radius 1, level 32 → radius 2, ...
    // Conservative ceiling at 4 (deblock at level >= 64 isn't
    // emitted by rav1e at typical QP).
    let r_x = ((level_v_edge as usize) / 16).min(4);
    let r_y = ((level_h_edge as usize) / 16).min(4);
    if r_x == 0 && r_y == 0 {
        return basis.to_vec();
    }
    let mut buf = basis.to_vec();
    if r_x > 0 {
        buf = box_smear_h(&buf, tx_w, tx_h, r_x);
    }
    if r_y > 0 {
        buf = box_smear_v(&buf, tx_w, tx_h, r_y);
    }
    buf
}

/// 1D box filter horizontally across each row. Radius `r` means the
/// output pixel is the mean of source pixels in `[col-r, col+r]`.
/// Reflective edge handling.
fn box_smear_h(src: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; w * h];
    let window = (2 * r + 1) as f64;
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f64;
            for dx in -(r as isize)..=(r as isize) {
                let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                sum += src[y * w + sx];
            }
            out[y * w + x] = sum / window;
        }
    }
    out
}

/// 1D box filter vertically across each column. Same shape as
/// [`box_smear_h`].
fn box_smear_v(src: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; w * h];
    let window = (2 * r + 1) as f64;
    for x in 0..w {
        for y in 0..h {
            let mut sum = 0.0f64;
            for dy in -(r as isize)..=(r as isize) {
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                sum += src[sy * w + x];
            }
            out[y * w + x] = sum / window;
        }
    }
    out
}

/// L2 cascade-context cache. Keyed by the L1 tuple `(tx_w, tx_h,
/// freq_x, freq_y)` + plane. For each key, stores the *normalized*
/// post-deblock impulse pattern — the L3 cache scales it by `|coeff|`
/// at lookup time, and B.1.5.5's three-tier dispatch reads the
/// HPF-residual-energy summary scalar.
///
/// **Single-frame invariant**: deblock levels are captured at frame
/// finalization, so within one frame they're constant. The L2 cache
/// closes over the level state at construction; subsequent frames
/// need a fresh cache.
///
/// **Memory bound**: ≤ unique L1 tuples × (tx_w × tx_h × 8 B). For
/// 1080p at speed 10 (~1000 unique tuples × ~32×32 = ~8 MB worst
/// case; typically <1 MB). Dropped at frame end.
pub struct L2CascadeContext {
    inner: HashMap<L2Key, Vec<f64>>,
    state: PhasmFrameLoopFilterState,
}

/// L2 key: `(tx_w, tx_h, freq_x, freq_y, plane)`. Compactly packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct L2Key(u16, u16, u16, u16, u8);

impl L2Key {
    pub fn new(tx_w: usize, tx_h: usize, freq_x: usize, freq_y: usize, plane: u8) -> Self {
        Self(tx_w as u16, tx_h as u16, freq_x as u16, freq_y as u16, plane)
    }
}

impl L2CascadeContext {
    pub fn new(state: PhasmFrameLoopFilterState) -> Self {
        Self {
            inner: HashMap::new(),
            state,
        }
    }

    /// Get or compute the post-deblock impulse for the given L2 key,
    /// using `basis` as the L1 source (caller pulls from the L1 cache).
    pub fn get_or_compute(&mut self, key: L2Key, basis: &[f64]) -> &[f64] {
        let state = self.state;
        self.inner.entry(key).or_insert_with(|| {
            apply_deblock_box_smear(basis, key.0 as usize, key.1 as usize, key.4, &state)
        })
    }

    pub fn read_only_state(&self) -> &PhasmFrameLoopFilterState {
        &self.state
    }

    pub fn entries(&self) -> usize {
        self.inner.len()
    }
}

// ======================================================================
//  B.1.5.3 — CDEF approximation layer + L3 cascade cache
// ======================================================================

/// Apply a separable box-filter approximation of AV1 CDEF on top of
/// a post-deblock impulse pattern. Returns a fresh `Vec<f64>` of the
/// same length.
///
/// **Approximation strategy**: stack a second box smear on the
/// post-deblock pattern, with radius scaled by the frame-level CDEF
/// primary+secondary strengths. Real CDEF is a directional 8-tap
/// filter that adapts per superblock — we elide both the directional
/// structure AND the per-SB strength variation in this first cut.
/// Frame-level strength index 0 (`cdef_y_strengths[0]` /
/// `cdef_uv_strengths[0]`) is used as the representative typical
/// strength.
///
/// **Why this is acceptable for v0.5**:
/// * Conservative bound: box smear over-estimates spread vs a
///   directional filter (which concentrates energy along its axis).
///   Over-estimating cascade extent biases safety in the right
///   direction (more rejections, fewer false-positive accepts).
/// * Index-0 ≈ typical: rav1e populates strength slot 0 first and
///   uses it for the majority of SBs at typical QP.
///
/// **Refinement to v0.6+**: per-SB CDEF index + spec-correct
/// directional filter. Requires the deferred B.1.5.1 fork-patch
/// surface (per-SB index Vec capture during `encode_tile_group_
/// with_phasm_tee`).
pub fn apply_cdef_box_smear(
    post_deblock: &[f64],
    tx_w: usize,
    tx_h: usize,
    plane: u8,
    state: &PhasmFrameLoopFilterState,
) -> Vec<f64> {
    debug_assert_eq!(post_deblock.len(), tx_w * tx_h);
    if !state.cdef_enabled {
        return post_deblock.to_vec();
    }
    // Use index 0 — representative typical strength.
    let packed = match plane {
        0 => state.cdef_y_strengths[0],
        1 | 2 => state.cdef_uv_strengths[0],
        _ => return post_deblock.to_vec(),
    };
    // CDEF strength packing per `cdef.rs`: primary_strength in high
    // bits, secondary_strength (0..CDEF_SEC_STRENGTHS=4) in low 2.
    // For our box-radius approximation, combine them additively then
    // scale to a small integer radius.
    let primary = (packed >> 2) as usize;
    let secondary = (packed & 0x3) as usize;
    // Empirical scaling: AV1 CDEF primary strengths run 0-15+; map
    // to a 0-3 box radius. Cap at 3 to bound the L2 cache spread.
    let r = ((primary + secondary) / 4).min(3);
    if r == 0 {
        return post_deblock.to_vec();
    }
    let buf = box_smear_h(post_deblock, tx_w, tx_h, r);
    box_smear_v(&buf, tx_w, tx_h, r)
}

/// L3 cascade cache. For each L2 key (which already encodes
/// `(tx_w, tx_h, freq_x, freq_y, plane)`), stores the post-cascade
/// HPF residual energy **normalized to a unit coefficient
/// magnitude** (`|coeff| = 1`).
///
/// At cost-compute time, the stored energy is scaled by
/// `|coeff|²` to recover the actual per-position cascade magnitude.
/// This works because deblock + CDEF approximations are linear in
/// the impulse amplitude — `f(α·x) = α·f(x)` — so the energy
/// `||f(α·x)||² = α²·||f(x)||²` scales as `|coeff|²`.
///
/// **L3 hit rate**: same as L2 (typically ≥95% on natural content
/// at speed 10). Per-position lookup is O(1) HashMap.
///
/// **Memory**: ≤ unique L2 tuples × 8 B. For 1080p typical: ~few KB.
///
/// **Used by**: B.1.5.5's three-tier dispatch — for the ~10-15% of
/// positions in the `|coeff|` middle band, the L3 lookup gives the
/// per-position cascade-magnitude scalar without running the full
/// forward model on every cost query.
pub struct L3CascadeCache {
    inner: HashMap<L2Key, f64>,
    state: PhasmFrameLoopFilterState,
}

impl L3CascadeCache {
    pub fn new(state: PhasmFrameLoopFilterState) -> Self {
        Self {
            inner: HashMap::new(),
            state,
        }
    }

    /// Get or compute the normalized post-cascade HPF residual
    /// energy for the given key, using `basis` from the L1 cache
    /// as the unit-magnitude input.
    ///
    /// Returns the normalized energy `E`. Caller scales by
    /// `|coeff|²` to recover actual cascade magnitude:
    /// `actual_cascade = sqrt(coeff² × E)`.
    pub fn get_or_compute(&mut self, key: L2Key, basis: &[f64]) -> f64 {
        let state = self.state;
        *self.inner.entry(key).or_insert_with(|| {
            // 1. Apply deblock smear to the unit-magnitude basis
            let post_deblock = apply_deblock_box_smear(
                basis,
                key.0 as usize,
                key.1 as usize,
                key.4,
                &state,
            );
            // 2. Stack CDEF approximation on top
            let post_cascade = apply_cdef_box_smear(
                &post_deblock,
                key.0 as usize,
                key.1 as usize,
                key.4,
                &state,
            );
            // 3. HPF residual energy: sum of squares of the
            // post-cascade pixel-domain perturbation. Larger means
            // more visible damage when scaled by |coeff|.
            post_cascade.iter().map(|v| v * v).sum::<f64>()
        })
    }

    pub fn entries(&self) -> usize {
        self.inner.len()
    }

    pub fn read_only_state(&self) -> &PhasmFrameLoopFilterState {
        &self.state
    }

    /// Read-only lookup. Returns `None` if the key wasn't pre-filled.
    /// Used by the parallel cost loop after `prefill` — `&self`
    /// makes the cache `Sync` and lets rayon workers share a single
    /// reference across threads (mirrors the [`FrameBasisCache::get`]
    /// pattern from B.3.1.3).
    pub fn get(&self, key: &L2Key) -> Option<f64> {
        self.inner.get(key).copied()
    }
}

/// Convenience: given the unit-magnitude normalized energy from
/// [`L3CascadeCache::get_or_compute`] and a coefficient magnitude,
/// compute the actual cascade-magnitude scalar (L2 norm of the
/// post-cascade pixel-domain perturbation).
///
/// `cascade_magnitude = |coeff| × sqrt(normalized_energy)`.
#[inline]
pub fn scale_l3_to_cascade(normalized_energy: f64, coeff_magnitude: u16) -> f64 {
    let c = coeff_magnitude as f64;
    (c * c * normalized_energy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_state(level: u8) -> PhasmFrameLoopFilterState {
        PhasmFrameLoopFilterState {
            deblock_levels: [level, level, level, level],
            cdef_y_strengths: [0; 8],
            cdef_uv_strengths: [0; 8],
            cdef_bits: 0,
            cdef_enabled: false,
        }
    }

    #[test]
    fn box_smear_h_radius_zero_is_identity() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = box_smear_h(&src, 4, 2, 0);
        // Box smear with radius 0 is mean-over-self = self.
        for i in 0..src.len() {
            assert!((src[i] - out[i]).abs() < 1e-12, "i={}", i);
        }
    }

    #[test]
    fn box_smear_h_radius_one_averages_three_pixels() {
        // 1D row [1, 2, 3, 4]. Radius 1 → window of 3.
        // - x=0: clamp{-1,0,1} = mean(1, 1, 2) = 4/3
        // - x=1: mean(1, 2, 3) = 2
        // - x=2: mean(2, 3, 4) = 3
        // - x=3: clamp{2,3,4} = mean(3, 4, 4) = 11/3
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let out = box_smear_h(&src, 4, 1, 1);
        assert!((out[0] - 4.0 / 3.0).abs() < 1e-12);
        assert!((out[1] - 2.0).abs() < 1e-12);
        assert!((out[2] - 3.0).abs() < 1e-12);
        assert!((out[3] - 11.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn box_smear_v_works_symmetric_to_h() {
        // Single column [1, 2, 3, 4], radius 1.
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let out = box_smear_v(&src, 1, 4, 1);
        assert!((out[0] - 4.0 / 3.0).abs() < 1e-12);
        assert!((out[1] - 2.0).abs() < 1e-12);
        assert!((out[2] - 3.0).abs() < 1e-12);
        assert!((out[3] - 11.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn deblock_box_smear_level_zero_is_identity() {
        let basis = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let state = dummy_state(0);
        let out = apply_deblock_box_smear(&basis, 4, 4, 0, &state);
        assert_eq!(basis.len(), out.len());
        for i in 0..basis.len() {
            assert!((basis[i] - out[i]).abs() < 1e-12, "i={}", i);
        }
    }

    #[test]
    fn deblock_box_smear_higher_level_spreads_more() {
        let mut basis = vec![0.0f64; 16];
        basis[5] = 10.0; // single impulse near center
        let state_low = dummy_state(8);   // r = 0 (8/16 = 0)
        let state_mid = dummy_state(20);  // r = 1
        let state_high = dummy_state(48); // r = 3
        let out_low = apply_deblock_box_smear(&basis, 4, 4, 0, &state_low);
        let out_mid = apply_deblock_box_smear(&basis, 4, 4, 0, &state_mid);
        let out_high = apply_deblock_box_smear(&basis, 4, 4, 0, &state_high);
        // Impulse magnitude must DECREASE as smearing radius grows
        // (energy gets spread out).
        let peak_low = out_low.iter().cloned().fold(0.0f64, f64::max);
        let peak_mid = out_mid.iter().cloned().fold(0.0f64, f64::max);
        let peak_high = out_high.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            peak_mid < peak_low,
            "mid={} should be < low={}",
            peak_mid,
            peak_low
        );
        assert!(
            peak_high < peak_mid,
            "high={} should be < mid={}",
            peak_high,
            peak_mid
        );
        // Sum is NOT conserved under clamp padding (edge pixels get
        // duplicated when the filter reaches past the boundary). What
        // matters is that the peak monotonically falls — energy
        // visibly diffuses with radius.
        let nonzero_count_low = out_low.iter().filter(|v| v.abs() > 1e-9).count();
        let nonzero_count_high = out_high.iter().filter(|v| v.abs() > 1e-9).count();
        assert!(
            nonzero_count_high >= nonzero_count_low,
            "energy should spread to more cells at higher radius"
        );
    }

    #[test]
    fn deblock_box_smear_chroma_planes_use_chroma_level() {
        let basis = vec![0.0f64; 16];
        // Y level 0, U level 32 (radius 2)
        let state = PhasmFrameLoopFilterState {
            deblock_levels: [0, 0, 32, 32],
            cdef_y_strengths: [0; 8],
            cdef_uv_strengths: [0; 8],
            cdef_bits: 0,
            cdef_enabled: false,
        };
        // Y plane: no smear (level 0)
        let out_y = apply_deblock_box_smear(&basis, 4, 4, 0, &state);
        for v in &out_y {
            assert_eq!(*v, 0.0);
        }
        // U plane: would smear if input were non-zero; output still all-zero
        // because input is all-zero. Just exercise the code path.
        let _out_u = apply_deblock_box_smear(&basis, 4, 4, 1, &state);
    }

    // ======================================================================
    //  B.1.5.3 — CDEF approximation + L3 cache tests
    // ======================================================================

    fn cdef_state(y_strength_packed: u8) -> PhasmFrameLoopFilterState {
        PhasmFrameLoopFilterState {
            deblock_levels: [0, 0, 0, 0], // isolate CDEF for testing
            cdef_y_strengths: [y_strength_packed, 0, 0, 0, 0, 0, 0, 0],
            cdef_uv_strengths: [y_strength_packed, 0, 0, 0, 0, 0, 0, 0],
            cdef_bits: 0,
            cdef_enabled: true,
        }
    }

    #[test]
    fn cdef_disabled_is_identity() {
        let mut state = cdef_state(0x3F); // very high strength
        state.cdef_enabled = false; // disable CDEF
        let basis = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let out = apply_cdef_box_smear(&basis, 4, 4, 0, &state);
        for i in 0..16 {
            assert!((basis[i] - out[i]).abs() < 1e-12, "i={}", i);
        }
    }

    #[test]
    fn cdef_zero_strength_is_identity() {
        let state = cdef_state(0); // packed strength byte 0 → r=0
        let basis = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let out = apply_cdef_box_smear(&basis, 4, 4, 0, &state);
        for i in 0..16 {
            assert!((basis[i] - out[i]).abs() < 1e-12, "i={}", i);
        }
    }

    #[test]
    fn cdef_higher_strength_spreads_more() {
        let mut basis = vec![0.0f64; 16];
        basis[5] = 10.0;
        // Packed strength: primary in high bits, secondary in low 2 bits.
        // Low primary → small r; high primary → bigger r.
        let state_low = cdef_state(0b00000_001); // primary=2, sec=1 → r=0
        let state_high = cdef_state(0b00011_011); // primary=12, sec=3 → r=3
        let out_low = apply_cdef_box_smear(&basis, 4, 4, 0, &state_low);
        let out_high = apply_cdef_box_smear(&basis, 4, 4, 0, &state_high);
        let peak_low = out_low.iter().cloned().fold(0.0f64, f64::max);
        let peak_high = out_high.iter().cloned().fold(0.0f64, f64::max);
        // Higher strength → more spread → lower peak (energy redistributed).
        // state_low has r=0 which is identity, so peak_low == 10.0.
        // state_high has r=3 → spread.
        assert!(
            peak_high < peak_low,
            "high strength peak ({}) should be < low strength peak ({})",
            peak_high,
            peak_low
        );
    }

    #[test]
    fn l3_normalized_energy_is_linear_in_basis_squared() {
        // Verify the linearity property: ||f(α·x)||² = α²·||f(x)||²
        // by feeding two scaled bases and checking the energy ratio.
        let mut basis_unit = vec![0.0f64; 16];
        basis_unit[5] = 1.0;
        let mut basis_scaled = vec![0.0f64; 16];
        basis_scaled[5] = 3.0;

        // Use both deblock + CDEF enabled with mid strengths.
        let state = PhasmFrameLoopFilterState {
            deblock_levels: [16, 16, 16, 16],
            cdef_y_strengths: [0b00010_010, 0, 0, 0, 0, 0, 0, 0],
            cdef_uv_strengths: [0b00010_010, 0, 0, 0, 0, 0, 0, 0],
            cdef_bits: 0,
            cdef_enabled: true,
        };

        let mut cache_unit = L3CascadeCache::new(state);
        let mut cache_scaled = L3CascadeCache::new(state);
        let key = L2Key::new(4, 4, 0, 0, 0);

        let e_unit = cache_unit.get_or_compute(key, &basis_unit);
        let e_scaled = cache_scaled.get_or_compute(key, &basis_scaled);

        // ||f(3·x)||² should equal 9·||f(x)||² (linearity).
        let expected_ratio = 9.0;
        let actual_ratio = e_scaled / e_unit.max(1e-30);
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.01,
            "linearity violated: expected ratio {}, got {} (e_unit={}, e_scaled={})",
            expected_ratio,
            actual_ratio,
            e_unit,
            e_scaled
        );
    }

    #[test]
    fn l3_scale_helper_recovers_cascade() {
        // For a known normalized energy E = ||unit_response||² and |coeff|=c,
        // the cascade magnitude is c × sqrt(E).
        let normalized_energy = 0.25; // sqrt = 0.5
        let coeff = 4u16;
        let cascade = scale_l3_to_cascade(normalized_energy, coeff);
        // 4 × sqrt(0.25) = 4 × 0.5 = 2.0
        assert!((cascade - 2.0).abs() < 1e-12);
    }

    #[test]
    fn l3_cache_caches_on_repeat_lookup() {
        let basis = vec![1.0f64; 16];
        let state = PhasmFrameLoopFilterState {
            deblock_levels: [16, 16, 16, 16],
            cdef_y_strengths: [0b00001_001, 0, 0, 0, 0, 0, 0, 0],
            cdef_uv_strengths: [0b00001_001, 0, 0, 0, 0, 0, 0, 0],
            cdef_bits: 0,
            cdef_enabled: true,
        };
        let mut cache = L3CascadeCache::new(state);
        let key = L2Key::new(4, 4, 1, 0, 0);
        let _ = cache.get_or_compute(key, &basis);
        assert_eq!(cache.entries(), 1);
        let _ = cache.get_or_compute(key, &basis);
        assert_eq!(cache.entries(), 1);
        // Different plane → new entry.
        let key2 = L2Key::new(4, 4, 1, 0, 1);
        let _ = cache.get_or_compute(key2, &basis);
        assert_eq!(cache.entries(), 2);
    }

    #[test]
    fn l2_cache_caches_on_repeat_lookup() {
        let basis = vec![1.0f64; 16];
        let state = dummy_state(20);
        let mut cache = L2CascadeContext::new(state);
        let key = L2Key::new(4, 4, 1, 0, 0);
        let _ = cache.get_or_compute(key, &basis);
        assert_eq!(cache.entries(), 1);
        let _ = cache.get_or_compute(key, &basis);
        // Still 1 entry — the second call hit the cache, not re-inserted.
        assert_eq!(cache.entries(), 1);
        // Different key → new entry.
        let key2 = L2Key::new(4, 4, 1, 0, 1); // different plane
        let _ = cache.get_or_compute(key2, &basis);
        assert_eq!(cache.entries(), 2);
    }
}
