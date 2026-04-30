// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Structural-invariant verification for encode-time CABAC stego
// (Phase 6D.9). Subsumes the deferred Q2 fingerprint study by
// measuring the four constructive properties from
// `cabac-bypass-bin-stego.md` Theorems 1–4 directly:
//
//   1. |coeff| histogram byte-identical between cover and stego
//      (sign-only domains; suffix-LSB domains shift by exactly
//      ±1 per modification).
//   2. |mvd| histogram byte-identical between cover and stego
//      (sign-only; same suffix caveat).
//   3. Sign-balance χ² shift in the sub-ppm range.
//   4. Slice-header / SPS / PPS byte-identical (verified
//      separately in the pipeline-level integration tests; the
//      orchestration layer doesn't touch them).
//
// Operates on `GopDecisionCache` pairs (cover and stego) so the
// validation stays at the orchestration layer — no encoder
// integration required.

use super::orchestrate::GopDecisionCache;

/// Histogram of magnitude bins: index = magnitude, value = count.
/// Capped at MAX_HIST_BIN; magnitudes ≥ that are clamped into
/// the last bin (rare for residual coeffs and MVD <128).
const MAX_HIST_BIN: usize = 256;

/// Magnitude histogram for residual coefficients across an entire
/// `GopDecisionCache`. Counts each nonzero coefficient once.
pub fn coeff_magnitude_histogram(cache: &GopDecisionCache) -> Vec<u64> {
    let mut hist = vec![0u64; MAX_HIST_BIN];
    for mb in &cache.mbs {
        for blk in &mb.residual_blocks {
            for &c in &blk.scan_coeffs {
                if c == 0 {
                    continue;
                }
                let bin = (c.unsigned_abs() as usize).min(MAX_HIST_BIN - 1);
                hist[bin] += 1;
            }
        }
    }
    hist
}

/// Magnitude histogram for MVDs across all MBs in the cache.
pub fn mvd_magnitude_histogram(cache: &GopDecisionCache) -> Vec<u64> {
    let mut hist = vec![0u64; MAX_HIST_BIN];
    for mb in &cache.mbs {
        for slot in &mb.mvd_slots {
            if slot.value == 0 {
                continue;
            }
            let bin = (slot.value.unsigned_abs() as usize).min(MAX_HIST_BIN - 1);
            hist[bin] += 1;
        }
    }
    hist
}

/// Per-domain sign-balance: (positive_count, negative_count) pairs.
/// Returned in (coeff, mvd) tuple; the χ² shift is computed
/// against expected 50/50.
pub fn sign_balance(cache: &GopDecisionCache) -> SignBalance {
    let mut sb = SignBalance::default();
    for mb in &cache.mbs {
        for blk in &mb.residual_blocks {
            for &c in &blk.scan_coeffs {
                if c == 0 {
                    continue;
                }
                if c > 0 {
                    sb.coeff_positive += 1;
                } else {
                    sb.coeff_negative += 1;
                }
            }
        }
        for slot in &mb.mvd_slots {
            if slot.value == 0 {
                continue;
            }
            if slot.value > 0 {
                sb.mvd_positive += 1;
            } else {
                sb.mvd_negative += 1;
            }
        }
    }
    sb
}

/// Sign-balance counts per domain.
#[derive(Default, Copy, Clone, Debug, Eq, PartialEq)]
pub struct SignBalance {
    pub coeff_positive: u64,
    pub coeff_negative: u64,
    pub mvd_positive: u64,
    pub mvd_negative: u64,
}

impl SignBalance {
    /// χ² statistic for coefficient sign balance vs uniform 50/50.
    /// Returns 0.0 if no coefficients present.
    pub fn coeff_chi_squared(&self) -> f64 {
        chi_squared(self.coeff_positive, self.coeff_negative)
    }

    /// χ² statistic for MVD sign balance vs uniform 50/50.
    pub fn mvd_chi_squared(&self) -> f64 {
        chi_squared(self.mvd_positive, self.mvd_negative)
    }

    /// Net coefficient sign delta as parts-per-million of the total
    /// signed count. Useful for quantifying "sub-ppm shift" claims.
    pub fn coeff_balance_ppm(&self) -> f64 {
        let total = (self.coeff_positive + self.coeff_negative) as f64;
        if total == 0.0 {
            return 0.0;
        }
        let delta = self.coeff_positive as i64 - self.coeff_negative as i64;
        (delta.unsigned_abs() as f64) * 1_000_000.0 / total
    }

    /// Net MVD sign delta as parts-per-million of the total signed count.
    pub fn mvd_balance_ppm(&self) -> f64 {
        let total = (self.mvd_positive + self.mvd_negative) as f64;
        if total == 0.0 {
            return 0.0;
        }
        let delta = self.mvd_positive as i64 - self.mvd_negative as i64;
        (delta.unsigned_abs() as f64) * 1_000_000.0 / total
    }
}

/// χ² statistic for a 2-cell expected-uniform distribution.
///
/// Uses the formula `((p - E)² + (n - E)²) / E` where `E = total/2`.
/// Since `p + n = total` and `(p - E) = -(n - E)`, this simplifies to
/// `2·(p - n)² / total` — i.e., **2× the textbook 1-dof Pearson χ²
/// of `(p - n)² / total`**. The 2× factor is consistent with the
/// Phase 3 CAVLC retrospective tooling, so we keep it for cross-
/// retrospective comparability. Anyone comparing to standard
/// critical values (e.g. 3.84 at p=0.05) should divide by 2.
fn chi_squared(pos: u64, neg: u64) -> f64 {
    let total = (pos + neg) as f64;
    if total == 0.0 {
        return 0.0;
    }
    let expected = total / 2.0;
    let p = pos as f64;
    let n = neg as f64;
    ((p - expected).powi(2) + (n - expected).powi(2)) / expected
}

/// Phase 6D.9 invariant report: cover ↔ stego comparison for one
/// orchestration round. Each field captures one of the four
/// structural invariants.
#[derive(Debug, Clone)]
pub struct InvariantReport {
    /// True iff |coeff| histograms are byte-identical (Theorem 1).
    /// Holds for sign-only domains; for suffix-LSB domains it holds
    /// up to a count shift equal to the modification count.
    pub coeff_magnitude_identical: bool,
    /// |coeff| histogram bin-by-bin diff. Useful when not identical
    /// (suffix-LSB case): each modification moves one count between
    /// adjacent bins.
    pub coeff_magnitude_diff: Vec<i64>,
    /// True iff |mvd| histograms are byte-identical.
    pub mvd_magnitude_identical: bool,
    pub mvd_magnitude_diff: Vec<i64>,
    /// Coefficient sign-balance χ² shift between cover and stego.
    pub coeff_chi_squared_delta: f64,
    /// MVD sign-balance χ² shift between cover and stego.
    pub mvd_chi_squared_delta: f64,
    /// Coefficient sign-balance shift in parts per million.
    pub coeff_ppm_shift: f64,
    /// MVD sign-balance shift in parts per million.
    pub mvd_ppm_shift: f64,
}

/// Compare a cover and stego decision cache, producing the
/// invariant report for Phase 6D.9 sign-off.
pub fn compare_invariants(
    cover: &GopDecisionCache,
    stego: &GopDecisionCache,
) -> InvariantReport {
    let coeff_cover_hist = coeff_magnitude_histogram(cover);
    let coeff_stego_hist = coeff_magnitude_histogram(stego);
    let coeff_diff: Vec<i64> = coeff_cover_hist
        .iter()
        .zip(coeff_stego_hist.iter())
        .map(|(c, s)| *s as i64 - *c as i64)
        .collect();
    let coeff_identical = coeff_diff.iter().all(|&d| d == 0);

    let mvd_cover_hist = mvd_magnitude_histogram(cover);
    let mvd_stego_hist = mvd_magnitude_histogram(stego);
    let mvd_diff: Vec<i64> = mvd_cover_hist
        .iter()
        .zip(mvd_stego_hist.iter())
        .map(|(c, s)| *s as i64 - *c as i64)
        .collect();
    let mvd_identical = mvd_diff.iter().all(|&d| d == 0);

    let cover_sb = sign_balance(cover);
    let stego_sb = sign_balance(stego);
    let coeff_chi_delta = (stego_sb.coeff_chi_squared() - cover_sb.coeff_chi_squared()).abs();
    let mvd_chi_delta = (stego_sb.mvd_chi_squared() - cover_sb.mvd_chi_squared()).abs();
    let coeff_ppm_shift = (stego_sb.coeff_balance_ppm() - cover_sb.coeff_balance_ppm()).abs();
    let mvd_ppm_shift = (stego_sb.mvd_balance_ppm() - cover_sb.mvd_balance_ppm()).abs();

    InvariantReport {
        coeff_magnitude_identical: coeff_identical,
        coeff_magnitude_diff: coeff_diff,
        mvd_magnitude_identical: mvd_identical,
        mvd_magnitude_diff: mvd_diff,
        coeff_chi_squared_delta: coeff_chi_delta,
        mvd_chi_squared_delta: mvd_chi_delta,
        coeff_ppm_shift,
        mvd_ppm_shift,
    }
}

/// Convert a [`GopDecisionCache`] from [`super::orchestrate`] usage —
/// any function consuming `mb_addr` for diagnostics needs to know.
/// This re-export keeps the validate API self-contained.
pub use super::orchestrate::MbResidualBlock;

#[allow(unused_imports)]
use super::Axis;

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::orchestrate::{
        pass1_collect_cover, pass2_stc_plan, pass3_apply_overrides, DomainMessages,
        MbDecision, MbResidualBlock as MbBlk, ResidualPathKind,
    };
    use super::super::MvdSlot;

    fn build_cover_gop() -> GopDecisionCache {
        let mut cache = GopDecisionCache::new();
        let mut scan = vec![0i32; 16];
        scan[0] = 5; scan[3] = -7; scan[6] = 2;
        cache.push(MbDecision {
            frame_idx: 0, mb_addr: 0,
            residual_blocks: vec![MbBlk {
                scan_coeffs: scan,
                start_idx: 0, end_idx: 15, ctx_block_cat: 1,
                path_kind: ResidualPathKind::Luma4x4 { block_idx: 0 },
            }],
            mvd_slots: vec![],
        });
        let mut scan = vec![0i32; 16];
        scan[1] = -4; scan[5] = 1;
        cache.push(MbDecision {
            frame_idx: 0, mb_addr: 1,
            residual_blocks: vec![MbBlk {
                scan_coeffs: scan,
                start_idx: 0, end_idx: 15, ctx_block_cat: 1,
                path_kind: ResidualPathKind::Luma4x4 { block_idx: 0 },
            }],
            mvd_slots: vec![
                MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 5 },
                MvdSlot { list: 0, partition: 0, axis: Axis::Y, value: -3 },
            ],
        });
        cache
    }

    #[test]
    fn no_op_stego_yields_byte_identical_invariants() {
        let cover = build_cover_gop();
        let stego = cover.clone();
        let report = compare_invariants(&cover, &stego);
        assert!(report.coeff_magnitude_identical);
        assert!(report.mvd_magnitude_identical);
        assert_eq!(report.coeff_chi_squared_delta, 0.0);
        assert_eq!(report.mvd_chi_squared_delta, 0.0);
    }

    #[test]
    fn sign_only_stego_preserves_magnitude_histogram() {
        // Three-pass with sign-only stego (CoeffSignBypass +
        // MvdSignBypass; suffix domains empty) — Theorem 1 says
        // magnitude histograms must be byte-identical.
        let cover = build_cover_gop();
        let cover_pass1 = pass1_collect_cover(&cover);

        // 1-bit message into the CoeffSignBypass cover (5 bits,
        // w=5, m=1).
        let messages = DomainMessages {
            coeff_sign_bypass: vec![1u8],
            ..Default::default()
        };
        let plan = pass2_stc_plan(&cover_pass1, &messages, 4).unwrap();
        let mut stego = cover.clone();
        pass3_apply_overrides(&mut stego, &cover_pass1.cover, &plan);

        let report = compare_invariants(&cover, &stego);
        assert!(
            report.coeff_magnitude_identical,
            "sign-only stego must preserve coeff magnitudes (diff={:?})",
            report.coeff_magnitude_diff,
        );
        assert!(report.mvd_magnitude_identical);
    }

    #[test]
    fn suffix_lsb_stego_changes_histogram_by_modification_count() {
        // Build cover with one large coefficient (|coeff|=20)
        // eligible for suffix LSB. Apply a forced flip (target=0,
        // cover=NOT 0=1). |coeff| goes 20 → 19. Magnitude histogram
        // bins shift.
        let mut cover = GopDecisionCache::new();
        let mut scan = vec![0i32; 16];
        scan[0] = 20;
        cover.push(MbDecision {
            frame_idx: 0, mb_addr: 0,
            residual_blocks: vec![MbBlk {
                scan_coeffs: scan, start_idx: 0, end_idx: 15, ctx_block_cat: 1,
                path_kind: ResidualPathKind::Luma4x4 { block_idx: 0 },
            }],
            mvd_slots: vec![],
        });
        let mut stego = cover.clone();
        // Manually apply the suffix-LSB flip (20 → 19).
        stego.mbs[0].residual_blocks[0].scan_coeffs[0] = 19;

        let report = compare_invariants(&cover, &stego);
        assert!(!report.coeff_magnitude_identical);
        // Bin 20 lost one, bin 19 gained one.
        assert_eq!(report.coeff_magnitude_diff[20], -1);
        assert_eq!(report.coeff_magnitude_diff[19], 1);
        // All other bins unchanged.
        for i in 0..report.coeff_magnitude_diff.len() {
            if i != 19 && i != 20 {
                assert_eq!(report.coeff_magnitude_diff[i], 0, "bin {i}");
            }
        }
    }

    #[test]
    fn sign_flips_shift_chi_squared_and_ppm() {
        // Cover: 4 coefficients, 2 positive 2 negative. Stego
        // flips one positive to negative → 3 negative + 1 positive.
        let mut cover = GopDecisionCache::new();
        let mut scan = vec![0i32; 16];
        scan[0] = 5; scan[1] = -3; scan[2] = 7; scan[3] = -2;
        cover.push(MbDecision {
            frame_idx: 0, mb_addr: 0,
            residual_blocks: vec![MbBlk {
                scan_coeffs: scan.clone(), start_idx: 0, end_idx: 15, ctx_block_cat: 1,
                path_kind: ResidualPathKind::Luma4x4 { block_idx: 0 },
            }],
            mvd_slots: vec![],
        });
        let mut stego = cover.clone();
        stego.mbs[0].residual_blocks[0].scan_coeffs[0] = -5;

        let report = compare_invariants(&cover, &stego);
        // Magnitudes preserved (sign flip).
        assert!(report.coeff_magnitude_identical);
        // χ² shift positive — cover was balanced (50/50), stego is
        // skewed (75% negative).
        assert!(report.coeff_chi_squared_delta > 0.0);
        // PPM shift: cover delta=0, stego delta=2/4 = 500_000ppm.
        assert!((report.coeff_ppm_shift - 500_000.0).abs() < 1.0);
    }

    #[test]
    fn coeff_magnitude_histogram_counts_correctly() {
        let cache = build_cover_gop();
        let hist = coeff_magnitude_histogram(&cache);
        // Coefficients: 5, -7, 2, -4, 1 → bins 5,7,2,4,1.
        assert_eq!(hist[1], 1);
        assert_eq!(hist[2], 1);
        assert_eq!(hist[4], 1);
        assert_eq!(hist[5], 1);
        assert_eq!(hist[7], 1);
        // Total = 5.
        let total: u64 = hist.iter().sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn mvd_magnitude_histogram_skips_zero_values() {
        let mut cache = GopDecisionCache::new();
        cache.push(MbDecision {
            frame_idx: 0, mb_addr: 0,
            residual_blocks: vec![],
            mvd_slots: vec![
                MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 0 },
                MvdSlot { list: 0, partition: 1, axis: Axis::X, value: 5 },
                MvdSlot { list: 0, partition: 2, axis: Axis::X, value: -10 },
            ],
        });
        let hist = mvd_magnitude_histogram(&cache);
        let total: u64 = hist.iter().sum();
        assert_eq!(total, 2, "value=0 must be excluded");
        assert_eq!(hist[5], 1);
        assert_eq!(hist[10], 1);
    }

    #[test]
    fn balanced_cover_has_zero_chi_squared() {
        let mut cache = GopDecisionCache::new();
        let mut scan = vec![0i32; 16];
        scan[0] = 5; scan[1] = -5;
        cache.push(MbDecision {
            frame_idx: 0, mb_addr: 0,
            residual_blocks: vec![MbBlk {
                scan_coeffs: scan, start_idx: 0, end_idx: 15, ctx_block_cat: 1,
                path_kind: ResidualPathKind::Luma4x4 { block_idx: 0 },
            }],
            mvd_slots: vec![],
        });
        let sb = sign_balance(&cache);
        assert_eq!(sb.coeff_chi_squared(), 0.0);
        assert_eq!(sb.coeff_balance_ppm(), 0.0);
    }
}
