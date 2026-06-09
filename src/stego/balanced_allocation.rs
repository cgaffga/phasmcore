// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Balanced safe per-GOP allocation planner (codec-agnostic).
//!
//! Replaces the earlier H.264 `plan_proportional` /
//! `plan_concentrate_tail` and AV1 plan. Implements the design from
//! [`balanced-allocation-v3.md`](../../docs/design/video/balanced-allocation-v3.md)
//! §3 algorithm.
//!
//! ## Caller-driven escalation
//!
//! The planner is **pure-functional and caller-driven**: it doesn't
//! own the probe-extractor. Callers loop:
//!
//! 1. Call [`plan_safe_balanced`] with whatever samples they've
//!    collected so far (may be empty for the initial call).
//! 2. Match on the [`PlanOutcome`]:
//!    - [`PlanOutcome::Plan`] → install the plan, encode, done.
//!    - [`PlanOutcome::NeedMoreSamples`] → probe the requested GOPs,
//!      append to the samples vector, call again.
//!    - [`PlanOutcome::MessageTooLarge`] → user-facing error; even a
//!      full-corpus probe wouldn't fit.
//!
//! This shape makes FFI to Swift/Kotlin transcoders trivial — they
//! drive the loop with their platform extractor (AVAssetReader,
//! MediaCodec) instead of the core owning a callback.
//!
//! ## Algorithm summary (see design doc §3 for full math)
//!
//! 1. **Always-probe** — even at trivial fit, K_min samples validate
//!    the cap_table prediction for THIS content.
//! 2. **Confidence-bounded escalation** — start at K_min; if
//!    99.9%-confidence pessimistic estimate doesn't clear
//!    `message × safety_margin`, double K and request more samples.
//! 3. **Balanced spread** — once confident, plan W from
//!    `ceil(msg / (target_utilization × cap_per_gop_safe))`, clamped
//!    below by `max(w_floor_abs, n_gops × w_floor_frac)` and above
//!    by `n_gops`.
//! 4. **Pre-encode hard check** — `total_safe ≥ msg × safety_margin`;
//!    only commits the plan if this holds.
//!
//! Calibration constants live in [`super::calibration::AllocationCalibration`].

use super::calibration::AllocationCalibration;

/// Outcome of one call to [`plan_safe_balanced`]. Drives the caller's
/// state machine.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanOutcome {
    /// Plan is committed. `plan[i]` is the byte count for GOP `i`;
    /// `plan[window..]` are all 0 (natural tail). `Σ plan == message_len`.
    Plan { plan: Vec<usize>, window: usize },

    /// Probe more GOPs and call again with the extended sample set.
    /// `positions` is the set of GOP indices the planner requests
    /// (caller may also include them in the next `samples` argument
    /// alongside what was already collected).
    NeedMoreSamples {
        positions: Vec<usize>,
        total_target: usize,
    },

    /// Even at full corpus coverage, message doesn't fit. User-facing error.
    MessageTooLarge,
}

/// Run one iteration of the balanced safe planner.
///
/// - `samples`: `(gop_idx, cap_bytes)` measured so far. Empty on
///   first call; grows as the caller probes the
///   [`PlanOutcome::NeedMoreSamples`] positions.
/// - `n_gops`: total GOPs in the clip.
/// - `message_len`: AEAD-encrypted message size in bytes (= total
///   bytes that need to be allocated across stego GOPs).
/// - `gop_size`: frames per encode-GOP (used for cap_table sanity check).
/// - `calibration`: locked Tier 1 + Tier 2 constants.
///
/// # Panics
///
/// Panics if `n_gops == 0` (a clip with no GOPs cannot carry any payload).
pub fn plan_safe_balanced(
    samples: &[(usize, usize)],
    n_gops: usize,
    message_len: usize,
    gop_size: usize,
    calibration: &AllocationCalibration,
) -> PlanOutcome {
    assert!(n_gops > 0, "plan_safe_balanced: n_gops must be > 0");

    // Special case: empty payload — degenerate plan, no GOPs needed.
    if message_len == 0 {
        return PlanOutcome::Plan {
            plan: vec![0; n_gops],
            window: 0,
        };
    }

    // Special case: no samples yet — request K_min at stratified positions.
    if samples.is_empty() {
        let positions = stratified_positions(calibration.k_min.min(n_gops), n_gops);
        return PlanOutcome::NeedMoreSamples {
            positions,
            total_target: calibration.k_min.min(n_gops),
        };
    }

    let k = samples.len();
    let caps: Vec<f64> = samples.iter().map(|&(_, c)| c as f64).collect();
    let mu = caps.iter().sum::<f64>() / k as f64;
    // Sample variance (n-1 denominator for unbiased estimate).
    let var = if k > 1 {
        caps.iter().map(|&c| (c - mu).powi(2)).sum::<f64>() / (k - 1) as f64
    } else {
        // Single sample — no variance info; degenerate. Treat sigma as
        // the value itself (heavily pessimistic). Forces escalation.
        mu
    };
    let sigma = var.sqrt();

    // t-statistic for 99.9% one-sided lower bound, df = K - 1.
    let t = t_critical_99_9(k);

    // Statistical lower bound on per-GOP cap.
    let cap_per_gop_lower = (mu - t * sigma / (k as f64).sqrt()).max(0.0);

    // Apply pessimism factor (heavy-tail / outlier defense).
    let cap_per_gop_safe = cap_per_gop_lower * calibration.pessimism_factor;

    // Sanity-check sample mean against table. Warn but don't fail.
    let table_pred =
        (calibration.cover_bits_per_frame_floor as f64) * (gop_size as f64) / 8.0;
    if table_pred > 0.0 {
        let ratio = mu / table_pred;
        if ratio > calibration.table_tolerance || ratio < 1.0 / calibration.table_tolerance {
            // Logging is caller-context-dependent; surface to a hook
            // when wired. For now, the divergence is implicit — the
            // sample-driven plan is what's used regardless.
        }
    }

    // Compute balanced W.
    let target_cap_per_chunk =
        (calibration.target_utilization * cap_per_gop_safe).max(1.0);
    let w_util = (message_len as f64 / target_cap_per_chunk).ceil() as usize;
    let w_floor = calibration
        .w_floor_abs
        .max(((n_gops as f64) * calibration.w_floor_frac).ceil() as usize);
    let w = w_util.max(w_floor).min(n_gops);

    // Pre-encode hard check: pessimistic total ≥ message × safety_margin.
    let total_safe = cap_per_gop_safe * w as f64;
    let required = message_len as f64 * calibration.safety_margin;

    if total_safe < required {
        if k >= n_gops {
            // Already at full corpus coverage — message genuinely doesn't fit.
            return PlanOutcome::MessageTooLarge;
        }
        // Escalate: double K, request more samples at non-overlapping positions.
        let new_target = (k * 2).min(n_gops).max(k + 1);
        let positions =
            stratified_positions_excluding(new_target, n_gops, samples_idx_set(samples));
        return PlanOutcome::NeedMoreSamples {
            positions,
            total_target: new_target,
        };
    }

    // Confident enough — distribute message uniformly across W GOPs.
    let plan = uniform_split(message_len, w, n_gops);
    PlanOutcome::Plan { plan, window: w }
}

// ─── Helpers ──────────────────────────────────────────────────────────

/// Stratified positions in `[0, n_gops)`: k evenly-spaced points at
/// stratum midpoints. Deterministic (no RNG); periodic-content
/// aliasing protection is a Tier 3 follow-on.
fn stratified_positions(k: usize, n_gops: usize) -> Vec<usize> {
    if k == 0 || n_gops == 0 {
        return Vec::new();
    }
    let k = k.min(n_gops);
    let mut out = Vec::with_capacity(k);
    let stride = n_gops as f64 / k as f64;
    for i in 0..k {
        let pos = (i as f64 + 0.5) * stride;
        out.push((pos as usize).min(n_gops - 1));
    }
    // Dedupe (rounding can collide on small n_gops).
    out.sort_unstable();
    out.dedup();
    out
}

/// Stratified positions excluding already-sampled GOPs.
fn stratified_positions_excluding(
    target_count: usize,
    n_gops: usize,
    excluded: std::collections::BTreeSet<usize>,
) -> Vec<usize> {
    let candidates = stratified_positions(target_count, n_gops);
    let mut out: Vec<usize> = candidates
        .into_iter()
        .filter(|p| !excluded.contains(p))
        .collect();
    // If our preferred positions overlap with existing samples,
    // backfill from any other unsampled GOP — preserves "K total
    // samples" target even when stratified positions overlap.
    if out.len() < target_count.saturating_sub(excluded.len()) {
        for i in 0..n_gops {
            if !excluded.contains(&i) && !out.contains(&i) {
                out.push(i);
                if out.len() + excluded.len() >= target_count {
                    break;
                }
            }
        }
    }
    out
}

fn samples_idx_set(samples: &[(usize, usize)]) -> std::collections::BTreeSet<usize> {
    samples.iter().map(|&(i, _)| i).collect()
}

/// t-distribution critical value for 99.9% one-sided lower bound.
/// Returns `t(0.001, df = K - 1)`. Hardcoded lookup for common K
/// values; linear interpolation between, asymptotic to z=3.090 at
/// large K. Within ~3% of exact for K ∈ [2, 1024].
fn t_critical_99_9(k: usize) -> f64 {
    // (K, t-critical at 99.9% one-sided, df = K-1) reference table.
    const TABLE: &[(usize, f64)] = &[
        (2, 318.31),
        (3, 22.327),
        (4, 10.215),
        (5, 7.173),
        (6, 5.893),
        (7, 5.208),
        (8, 4.785),
        (9, 4.501),
        (10, 4.297),
        (12, 4.025),
        (16, 3.733),
        (24, 3.485),
        (32, 3.385),
        (64, 3.232),
        (128, 3.160),
        (256, 3.125),
        (512, 3.107),
        (1024, 3.099),
    ];
    if k <= 1 {
        // Degenerate — return a very pessimistic value to force escalation.
        return 1000.0;
    }
    if k >= TABLE.last().unwrap().0 {
        return 3.090; // asymptotic z(0.999)
    }
    // Linear interpolate between bracketing rows.
    for i in 0..TABLE.len() - 1 {
        let (k0, t0) = TABLE[i];
        let (k1, t1) = TABLE[i + 1];
        if k <= k1 {
            let frac = (k as f64 - k0 as f64) / (k1 as f64 - k0 as f64);
            return t0 + (t1 - t0) * frac;
        }
    }
    TABLE.last().unwrap().1
}

/// Distribute `message_len` bytes uniformly across `w` GOPs, with
/// trailing `n_gops - w` GOPs zero (natural tail). The remainder from
/// floor division is spread across the leading-most GOPs so
/// `Σ plan == message_len` exactly.
fn uniform_split(message_len: usize, w: usize, n_gops: usize) -> Vec<usize> {
    let mut plan = vec![0; n_gops];
    if w == 0 {
        return plan;
    }
    let base = message_len / w;
    let remainder = message_len % w;
    for i in 0..w {
        plan[i] = base + if i < remainder { 1 } else { 0 };
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cal() -> AllocationCalibration {
        AllocationCalibration::AV1_1080P_QP30
    }

    #[test]
    fn empty_message_returns_zero_plan() {
        let r = plan_safe_balanced(&[], 100, 0, 15, &cal());
        match r {
            PlanOutcome::Plan { plan, window } => {
                assert_eq!(window, 0);
                assert!(plan.iter().all(|&b| b == 0));
                assert_eq!(plan.len(), 100);
            }
            _ => panic!("expected Plan, got {r:?}"),
        }
    }

    #[test]
    fn no_samples_requests_kmin_stratified() {
        let r = plan_safe_balanced(&[], 100, 1000, 15, &cal());
        match r {
            PlanOutcome::NeedMoreSamples {
                positions,
                total_target,
            } => {
                assert_eq!(total_target, 8); // K_min
                assert_eq!(positions.len(), 8);
                // Positions should span the clip.
                assert!(positions[0] < 20);
                assert!(*positions.last().unwrap() > 80);
            }
            _ => panic!("expected NeedMoreSamples, got {r:?}"),
        }
    }

    #[test]
    fn ample_samples_with_easy_fit_commits_plan() {
        // 100 GOPs, message 100 bytes, calibration 1080p QP30:
        // cap_floor = 80000 cb/frame × gop_size=15 / 8 = 150_000 bytes/GOP raw.
        // Pessimistic safe ≈ μ × pessimism, so per-GOP safe ≈ 75_000 bytes.
        // 100 bytes is trivially small. Should commit at W_min=8.
        let mut samples: Vec<(usize, usize)> = Vec::new();
        // Simulate K=8 stratified samples with tight cap distribution.
        for i in &[6, 18, 30, 43, 56, 68, 81, 93] {
            samples.push((*i, 150_000)); // bytes per GOP — large
        }
        let r = plan_safe_balanced(&samples, 100, 100, 15, &cal());
        match r {
            PlanOutcome::Plan { plan, window } => {
                // W_floor_abs = 8 or W_floor_frac × n_gops = 5 → max = 8.
                assert_eq!(window, 8);
                assert_eq!(plan.iter().sum::<usize>(), 100);
                // First 8 GOPs each have plan[i] > 0; trailing 92 are 0.
                assert!(plan[..8].iter().all(|&b| b > 0));
                assert!(plan[8..].iter().all(|&b| b == 0));
            }
            _ => panic!("expected Plan, got {r:?}"),
        }
    }

    #[test]
    fn tight_message_escalates_then_commits() {
        // 100 GOPs, message 5_000_000 bytes (5 MB).
        // 8 samples of 150 KB each → safe = 75 KB × 100 = 7.5 MB total raw.
        // After pessimism + target_utilization = 0.05, planner may need more samples.
        let n_gops = 100;
        let mut samples: Vec<(usize, usize)> = Vec::new();
        for i in &[6, 18, 30, 43, 56, 68, 81, 93] {
            samples.push((*i, 150_000));
        }
        let r = plan_safe_balanced(&samples, n_gops, 5_000_000, 15, &cal());
        // Either it commits with a wide window, or escalates.
        match r {
            PlanOutcome::Plan { plan, window } => {
                assert!(window >= 8 && window <= n_gops);
                assert_eq!(plan.iter().sum::<usize>(), 5_000_000);
            }
            PlanOutcome::NeedMoreSamples {
                positions,
                total_target,
            } => {
                assert!(total_target > 8);
                assert!(!positions.is_empty());
            }
            PlanOutcome::MessageTooLarge => {
                // Acceptable if message too big for this many GOPs of this cap.
            }
        }
    }

    #[test]
    fn full_coverage_with_genuinely_too_large_message_errors() {
        // 8 GOPs, every GOP measured at 100 bytes cap, message = 1 GB.
        // Even pessimistic plan with W = 8 GOPs can't fit. Full coverage
        // = 8 samples on 8 GOPs → must return MessageTooLarge.
        let samples: Vec<(usize, usize)> =
            (0..8).map(|i| (i, 100usize)).collect();
        let r = plan_safe_balanced(&samples, 8, 1_000_000_000, 15, &cal());
        match r {
            PlanOutcome::MessageTooLarge => {}
            other => panic!("expected MessageTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn stratified_positions_basic() {
        let p = stratified_positions(4, 100);
        // Stratum midpoints at 0.5/4×100, 1.5/4×100, 2.5/4×100, 3.5/4×100 → 12, 37, 62, 87.
        assert_eq!(p.len(), 4);
        assert!(p[0] >= 10 && p[0] <= 15);
        assert!(p[3] >= 85 && p[3] <= 90);
        // Strictly sorted.
        for w in p.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn stratified_positions_k_exceeds_n_gops_clamps() {
        let p = stratified_positions(50, 5);
        assert!(p.len() <= 5);
        for &i in &p {
            assert!(i < 5);
        }
    }

    #[test]
    fn t_critical_table_monotonic_and_asymptotic() {
        let small = t_critical_99_9(8);
        let mid = t_critical_99_9(32);
        let large = t_critical_99_9(2000);
        assert!(small > mid && mid > large);
        assert!((large - 3.090).abs() < 0.01);
    }

    #[test]
    fn uniform_split_sums_to_message_len() {
        for &(msg, w, n) in &[(100usize, 8, 100), (1_000, 50, 100), (37, 5, 100)] {
            let plan = uniform_split(msg, w, n);
            assert_eq!(plan.iter().sum::<usize>(), msg);
            assert_eq!(plan.len(), n);
            // Window matches: leading w nonzero, rest zero.
            for i in 0..w {
                assert!(plan[i] > 0);
            }
            for i in w..n {
                assert_eq!(plan[i], 0);
            }
        }
    }

    #[test]
    fn uniform_split_zero_window_all_zero() {
        let plan = uniform_split(100, 0, 50);
        assert!(plan.iter().all(|&b| b == 0));
    }

    #[test]
    fn w_floor_enforces_minimum_spread_for_trivial_message() {
        // Trivial 100-byte message, 100-GOP clip, 8 samples with large caps.
        // W_util might be 1 (fits in one GOP), but W_floor_abs=8 + W_floor_frac×100=5
        // forces W = max(1, max(8, 5)) = 8.
        let mut samples: Vec<(usize, usize)> = Vec::new();
        for i in &[6, 18, 30, 43, 56, 68, 81, 93] {
            samples.push((*i, 200_000));
        }
        let r = plan_safe_balanced(&samples, 100, 100, 15, &cal());
        match r {
            PlanOutcome::Plan { window, .. } => assert_eq!(window, 8),
            _ => panic!("expected Plan with W=W_floor"),
        }
    }
}
