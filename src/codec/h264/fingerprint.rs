// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 encoder fingerprint extraction.
//!
//! Phase 6.0c — statistical feature extractor for the encoder-
//! fingerprint regression test. Parses an H.264 MP4 and tallies
//! syntax-level histograms that let a classifier distinguish
//! "this clip was encoded by encoder A vs encoder B vs phasm" on
//! the basis of the bitstream-level choices each encoder makes
//! (partition-size distribution, QP deltas, skip-run lengths, etc.).
//!
//! Used by:
//!   - `docs/design/video/h264/encoder-algorithms/fingerprint-regression.md`
//!   - `core/tests/h264_fingerprint_regression.rs` (Phase 6A+)
//!
//! **Intentionally NOT gated behind `h264-encoder`**: we need this
//! for third-party clip analysis even in the CLI binary build that
//! ships no encoder. It's a decode/analysis tool, not an encoder.
//!
//! Scope is kept small (~30 scalar features). See the design doc
//! for the full feature-by-feature motivation. Pixel content is
//! deliberately out of scope — that's the oracle harness's job.

use super::H264Error;

/// Histogram size for `mb_type` — H.264 defines fewer than 32 values
/// for I and P slices combined (§ 7.4.5 Tables 7-11 / 7-13).
const MB_MODE_BINS: usize = 32;

/// Histogram size for `coded_block_pattern` — 48 distinct CBP values
/// for yuv420p Intra/Inter (§ 9.1.2 Table 9-4).
const CBP_BINS: usize = 48;

/// Histogram size for `mb_qp_delta` signed — QP is 0–51, delta ranges
/// ±25; we use 52 bins covering the full reachable range.
const QP_DELTA_BINS: usize = 52;

/// Histogram size for non-zero coefficient count — 4×4 block has up
/// to 16 non-zero coeffs (§ 9.2.1 Table 9-5).
const NZC_BINS: usize = 17;

/// Histogram size for log2-bucketed MV magnitude — 12 bins cover
/// |MVD| from 0 to 2^11 quarter-pels.
const MV_MAG_BINS: usize = 12;

/// Encoder-fingerprint feature vector.
///
/// Extracted from a single MP4 containing an H.264 Baseline/Main/High
/// stream. All histograms are raw counts (not normalized); downstream
/// classification normalizes by stream length.
#[derive(Debug, Clone, PartialEq)]
pub struct Fingerprint {
    /// `mb_type` histogram across every macroblock in every slice.
    pub mb_mode_hist: [u32; MB_MODE_BINS],
    /// `coded_block_pattern` histogram (yuv420p).
    pub cbp_hist: [u32; CBP_BINS],
    /// `mb_qp_delta` (signed) histogram, offset by +25 so bin 25 == 0.
    pub qp_delta_hist: [u32; QP_DELTA_BINS],
    /// Non-zero-coefficient-count histogram across 4×4 luma blocks.
    pub nzc_hist: [u32; NZC_BINS],
    /// Log2-bucketed |MVD| histogram (bin `k` = 2^k ≤ |mvd| < 2^(k+1)).
    pub mv_mag_hist: [u32; MV_MAG_BINS],
    /// Fraction of inter-MBs whose MVD is exactly (0, 0).
    pub mv_zero_ratio: f32,
    /// Bitmask of which optional slice-header fields were present:
    /// bit 0 = num_ref_idx_active_override_flag,
    /// bit 1 = direct_spatial_mv_pred_flag,
    /// bit 2 = cabac_init_idc present,
    /// bit 3 = slice_qp_delta nonzero,
    /// bit 4 = disable_deblocking_filter_idc != 0,
    /// bit 5 = slice_alpha_c0_offset_div2 nonzero,
    /// bit 6 = slice_beta_offset_div2 nonzero.
    pub slice_flags: u16,
    /// SPS constraint_setX_flag bits (6 bits, § 7.4.2.1.1).
    pub sps_constraints: u8,
    /// 0 = CAVLC, 1 = CABAC (PPS `entropy_coding_mode_flag`).
    pub entropy_mode: u8,
    /// Average I-frame interval (0 if fewer than 2 I-frames observed).
    pub gop_interval: u32,
    /// `num_ref_idx_l0_active_minus1 + 1` from the PPS.
    pub ref_list_len: u8,
    /// Total macroblocks seen — used for normalization downstream.
    pub total_mbs: u32,
    /// Total slices seen — sanity check.
    pub total_slices: u32,
}

impl Default for Fingerprint {
    fn default() -> Self {
        // Manual impl — Rust's derive doesn't emit Default for arrays
        // longer than 32 elements, so we spell out the zeros.
        Self {
            mb_mode_hist: [0; MB_MODE_BINS],
            cbp_hist: [0; CBP_BINS],
            qp_delta_hist: [0; QP_DELTA_BINS],
            nzc_hist: [0; NZC_BINS],
            mv_mag_hist: [0; MV_MAG_BINS],
            mv_zero_ratio: 0.0,
            slice_flags: 0,
            sps_constraints: 0,
            entropy_mode: 0,
            gop_interval: 0,
            ref_list_len: 0,
            total_mbs: 0,
            total_slices: 0,
        }
    }
}

impl Fingerprint {
    /// Flatten into a feature vector for classifier input.
    ///
    /// Histograms are normalized by `total_mbs` so the result is a
    /// probability distribution per feature group. Order matters:
    /// downstream classifier training assumes this exact layout.
    pub fn to_vec(&self) -> Vec<f32> {
        let n = self.total_mbs.max(1) as f32;
        let mut v = Vec::with_capacity(
            MB_MODE_BINS + CBP_BINS + QP_DELTA_BINS + NZC_BINS + MV_MAG_BINS + 8,
        );
        v.extend(self.mb_mode_hist.iter().map(|&x| x as f32 / n));
        v.extend(self.cbp_hist.iter().map(|&x| x as f32 / n));
        v.extend(self.qp_delta_hist.iter().map(|&x| x as f32 / n));
        v.extend(self.nzc_hist.iter().map(|&x| x as f32 / n));
        v.extend(self.mv_mag_hist.iter().map(|&x| x as f32 / n));
        v.push(self.mv_zero_ratio);
        v.push(self.slice_flags as f32);
        v.push(self.sps_constraints as f32);
        v.push(self.entropy_mode as f32);
        v.push(self.gop_interval as f32);
        v.push(self.ref_list_len as f32);
        v.push(self.total_mbs as f32);
        v.push(self.total_slices as f32);
        v
    }
}

/// Extract a fingerprint from an H.264 MP4.
///
/// Phase 6.0c ships the scaffolding and data model only. The
/// per-field extraction lands as each Phase 6A/6B sub-phase makes
/// the corresponding parse state accessible:
///
///   - SPS/PPS/slice-header fields: already parseable today, filled
///     in during Phase 6A.5 (bitstream writer + SPS/PPS mirror).
///   - `mb_mode_hist`, `cbp_hist`: Phase 6A.3 (intra mode decision
///     surface) + Phase 6A.4 (CAVLC output surface).
///   - `qp_delta_hist`: Phase 6A.9 (adaptive QP produces delta).
///   - `nzc_hist`: Phase 6A.4 (CAVLC residual path).
///   - `mv_mag_hist`, `mv_zero_ratio`: Phase 6B.2 (ME pass).
///   - `gop_interval`: Phase 6B.4 (DPB + GOP structure).
///
/// Until then, this returns `Err(NotYetImplemented)`; the
/// regression test stays inactive.
pub fn extract(_mp4_bytes: &[u8]) -> Result<Fingerprint, H264Error> {
    Err(H264Error::Unsupported(
        "fingerprint::extract — Phase 6.0c scaffold, populated by Phase 6A/6B sub-phases"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_to_vec_length_stable() {
        let fp = Fingerprint::default();
        let v = fp.to_vec();
        // Histograms + scalars — pinning this length keeps downstream
        // classifier weights pegged to a stable feature-vector shape.
        assert_eq!(
            v.len(),
            MB_MODE_BINS + CBP_BINS + QP_DELTA_BINS + NZC_BINS + MV_MAG_BINS + 8,
        );
    }

    #[test]
    fn fingerprint_default_vec_is_finite() {
        let fp = Fingerprint::default();
        assert!(fp.to_vec().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn extract_returns_not_yet_impl() {
        let result = extract(b"not a real mp4");
        assert!(matches!(result, Err(H264Error::Unsupported(_))));
    }
}
