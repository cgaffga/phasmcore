// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! §6E-A6.0 — B-slice macroblock mode decision (encoder side).
//!
//! Driven by the algorithm note at
//! `docs/design/h264-encoder-algorithms/6E-A6-bslice-partitions.md`.
//!
//! ## Sub-phase activation
//!
//! Each B mb_type variant lights up in a specific sub-phase. This
//! module defines the [`BMbDecision`] type and the `mb_decision_b`
//! entry point as the dispatch surface; the body progressively
//! enlarges its candidate set across the sub-phases:
//!
//! | Sub-phase | mb_types added | `BMbDecision` variants |
//! |---|---|---|
//! | §6E-A4(c)-lite (shipped) | 0, B_Skip | `Skip`, `Direct16x16` |
//! | §6E-A6.1 | 1, 2, 3 | `L0_16x16`, `L1_16x16`, `Bi_16x16` |
//! | §6E-A6.2 | 4..21 | `Partitioned` (shape + parts via `b_partitioned`) |
//! | §6E-A6.3 | 22 (sub_mb_type 0..3) | `B8x8` |
//! | §6E-A6.4 | 22 (sub_mb_type 4..12) | `B8x8` extended |
//!
//! The §6E-A6.0 stub returns only `Skip` or `Direct16x16` —
//! preserving §6E-A4(c)-lite's emission shape exactly. As §6E-A6.1
//! / .2 / .3 land, they progressively widen the candidate set.
//!
//! ## Mode-decision strategy
//!
//! Cost-based selection over the candidate set with mode-preference
//! penalties borrowed from ffmpeg's `ff_estimate_b_frame_motion`
//! pattern (codec-agnostic, allowed per `memory/h264_clean_room_audit.md`):
//!
//! - Direct  ×0 (free; chosen when neighbours agree on a clean
//!            predictor)
//! - Bi      ×1 (slight penalty)
//! - L1      ×2
//! - L0      ×3
//! - Skip    handled via a separate up-front check (CBP-zero shortcut)
//!
//! See `docs/design/h264-encoder-algorithms/6E-A6-bslice-partitions.md`
//! § "ffmpeg reference" for the rationale + the empirical x264-medium
//! distribution we calibrate against in §6E-A6.5.

use super::motion_estimation::MotionVector;
use super::partition_state::EncoderMvGrid;

/// §6E-A6.1 / §6E-A6.2 — process-wide Mutex for tests that
/// manipulate the `PHASM_B_FORCE_MODE` env var. Tests touching
/// the env var (force-mode round-trip + distribution-match)
/// must hold this lock to prevent parallel-test races.
#[cfg(test)]
pub(crate) static B_FORCE_MODE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Outcome of a B-slice mode decision for a single macroblock.
///
/// One variant per spec mb_type family. As §6E-A6.1..§6E-A6.3 land,
/// new variants get filled in with the partition-specific data the
/// encoder needs to emit + populate the MV grid.
#[derive(Debug, Clone)]
pub enum BMbDecision {
    /// `B_Skip` — `mb_skip_flag = 1`, no further syntax. Decoder
    /// derives the MV via spatial-direct.
    Skip,

    /// `mb_type = 0` (`B_Direct_16x16`) — `mb_skip_flag = 0`,
    /// `mb_type = 0`, `cbp_value = 0`, no MVDs, no residual. Decoder
    /// derives the MV spatially.
    Direct16x16,

    /// `mb_type = 1` (`B_L0_16x16`). One L0 MV, no L1. Single-ref
    /// configuration locks `ref_idx_l0 = 0` (not emitted on the
    /// wire — `num_ref_idx_l0_active_minus1 = 0` per §6E-A4 SPS).
    L0_16x16 { mv: MotionVector },

    /// `mb_type = 2` (`B_L1_16x16`). One L1 MV, no L0. `ref_idx_l1 = 0`.
    L1_16x16 { mv: MotionVector },

    /// `mb_type = 3` (`B_Bi_16x16`). One L0 MV + one L1 MV.
    /// `ref_idx_l0 = ref_idx_l1 = 0`.
    Bi_16x16 { mv_l0: MotionVector, mv_l1: MotionVector },

    /// §6E-A6.2 — partitioned B mb_type. Covers all 18 variants
    /// 4..21 (both 16x8 and 8x16 shapes; the shape is encoded in the
    /// looked-up `BPartitionedMeta`). `mb_type` is the spec value
    /// (4..=21); `parts` is the per-partition MV pair, where each
    /// partition's `mv_l0` / `mv_l1` is `Some` only when the
    /// partition's list usage from spec Table 7-14 includes that
    /// list (per `partitioned_b_meta(mb_type).part0/part1`).
    Partitioned {
        mb_type: u8,
        parts: [super::b_partitioned::BPartitionMv; 2],
    },

    /// `mb_type = 22` (`B_8x8`). Four sub-MBs each with their own
    /// `sub_mb_type`. **Lands in §6E-A6.3** (sub_mb_types 0..3 only,
    /// per the x264-medium distribution finding).
    #[allow(dead_code)]
    B8x8 {/* sub_mb_types: [u8; 4], parts: [Vec<PartitionMv>; 4] */},
}

/// §6E-A6.0 — entry point for B-slice MB mode decision. The default
/// implementation matches §6E-A4(c)-lite output exactly: deterministic
/// ~50/50 hash-based mix of `Skip` and `Direct16x16`.
///
/// **§6E-A6.1+ widens this** progressively: L0/L1/Bi 16x16 (§6E-A6.1),
/// partitioned 16x8/8x16 (§6E-A6.2), B_8x8 (§6E-A6.3). Real cost-
/// based selection (ffmpeg-MPEG-style ×0/×1/×2/×3 penalties) lands
/// alongside ME extensions.
///
/// **Test override (§6E-A6.1)**: when the `PHASM_B_FORCE_MODE` env
/// var is set, all B-MBs in the frame return the forced decision
/// instead of the hash-mix. Recognized values:
/// - `skip`        → `Skip`
/// - `direct`      → `Direct16x16`
/// - `l0_16x16`    → `L0_16x16 { mv: (0, 0) }`
/// - `l1_16x16`    → `L1_16x16 { mv: (0, 0) }`
/// - `bi_16x16`    → `Bi_16x16 { mv_l0 = mv_l1 = (0, 0) }`
///
/// Test-only — production callers should never set the var. The
/// override path is what `walk_b_l0_16x16` / `walk_b_l1_16x16` /
/// `walk_b_bi_16x16` round-trip tests use to exercise non-direct
/// paths before real ME-based mode selection lands.
///
/// `_grid` / `_mb_x` / `_mb_y` are accepted now so the signature
/// is forward-stable across the §6E-A6.x sub-phases.
pub fn mb_decision_b(
    grid: &EncoderMvGrid,
    mb_x: usize,
    mb_y: usize,
    frame_num: u32,
    mb_addr: u32,
) -> BMbDecision {
    // §6E-A6.1 test-only override.
    if let Some(forced) = forced_b_mode_from_env() {
        return forced;
    }

    // §6E-A6.1 distribution-match path. Hash-based bucket selection
    // matching the x264 medium-CRF23 reference distribution (per
    // `core/tests/data/x264_reference_mb_histogram.txt`):
    //
    //   ~50% Skip, ~35% Direct16x16, ~6% L0_16x16, ~5% L1_16x16,
    //   ~4% Bi_16x16
    //
    // Skip + Direct still dominate; the new L0/L1/Bi 16x16 modes
    // appear in the right proportions to make the L3 mb_type
    // fingerprint align with x264. Magnitudes are zero-MVD (i.e.
    // mv == spatial-direct predictor) for the §6E-A6.1 ship; real
    // ME + non-zero MVDs land in a follow-up alongside CBP-non-zero
    // residual emission.
    let bucket = mb_decision_bucket(frame_num, mb_addr);
    match bucket {
        // 0..=49   → Skip   (50%)
        0..=49 => BMbDecision::Skip,
        // 50..=84  → Direct (35%)
        50..=84 => BMbDecision::Direct16x16,
        // 85..=90  → L0_16x16 (6%)
        85..=90 => {
            let mv = predict_b_partition_mv_l0(grid, mb_x, mb_y);
            BMbDecision::L0_16x16 { mv }
        }
        // 91..=95  → L1_16x16 (5%)
        91..=95 => {
            let mv = predict_b_partition_mv_l1(grid, mb_x, mb_y);
            BMbDecision::L1_16x16 { mv }
        }
        // 96..=99  → Bi_16x16 (4%)
        _ => {
            let mv_l0 = predict_b_partition_mv_l0(grid, mb_x, mb_y);
            let mv_l1 = predict_b_partition_mv_l1(grid, mb_x, mb_y);
            BMbDecision::Bi_16x16 { mv_l0, mv_l1 }
        }
    }
}

/// Hash-based bucket [0, 100) for §6E-A6.1 mode-mix selection.
/// Same hash family as `mb_skip_or_direct_decision` for stability,
/// just spread across 100 buckets instead of 2. Determinism: pure
/// function of (frame_num, mb_addr) so encoder + walker (via the
/// same call) agree on the chosen mode.
fn mb_decision_bucket(frame_num: u32, mb_addr: u32) -> u32 {
    let mut x = (frame_num as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = x.wrapping_add(mb_addr as u64);
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    (x % 100) as u32
}

/// Predicted L0 MV for a B-MB's 16x16 partition — used as the
/// chosen MV for §6E-A6.1 zero-MVD mode emission. (Real ME
/// against L1 reference + cost-based MV refinement lands in
/// a follow-up.)
fn predict_b_partition_mv_l0(grid: &EncoderMvGrid, mb_x: usize, mb_y: usize) -> MotionVector {
    super::partition_state::predict_mv_for_mb_partition(
        grid, mb_x * 4, mb_y * 4,
        /* part_w_4x4 */ 4, /* part_h_4x4 */ 4,
        /* mb_part_idx */ 0, /* current_ref_idx */ 0,
    )
}

/// Mirror of [`predict_b_partition_mv_l0`] for List 1.
fn predict_b_partition_mv_l1(grid: &EncoderMvGrid, mb_x: usize, mb_y: usize) -> MotionVector {
    super::b_direct_predictor::predict_mv_for_partition_l1_pub(
        grid, mb_x * 4, mb_y * 4, /* current_ref_idx */ 0,
    )
}

/// §6E-A6.1 — read `PHASM_B_FORCE_MODE` env var and translate to a
/// `BMbDecision`. Returns `None` if the var is unset / unrecognized
/// (production path falls through to the §6E-A4(c)-lite hash mix).
fn forced_b_mode_from_env() -> Option<BMbDecision> {
    let var = std::env::var("PHASM_B_FORCE_MODE").ok()?;
    let zero = MotionVector { mv_x: 0, mv_y: 0 };
    match var.to_ascii_lowercase().as_str() {
        "skip" => Some(BMbDecision::Skip),
        "direct" => Some(BMbDecision::Direct16x16),
        "l0_16x16" => Some(BMbDecision::L0_16x16 { mv: zero }),
        "l1_16x16" => Some(BMbDecision::L1_16x16 { mv: zero }),
        "bi_16x16" => Some(BMbDecision::Bi_16x16 {
            mv_l0: zero,
            mv_l1: zero,
        }),
        // §6E-A6.2 — partitioned variants. `partitioned_<mb_type>`
        // forces a specific 4..21 mb_type for round-trip testing.
        s if s.starts_with("partitioned_") => {
            let mb_type: u8 = s["partitioned_".len()..].parse().ok()?;
            forced_partitioned_decision(mb_type)
        }
        _ => None,
    }
}

/// §6E-A6.2 — build a zero-MV `Partitioned` decision for a given
/// mb_type (4..=21). Used by the env-var override + tests.
fn forced_partitioned_decision(mb_type: u8) -> Option<BMbDecision> {
    use super::b_partitioned::{partitioned_b_meta, BListUse, BPartitionMv};
    let zero = MotionVector { mv_x: 0, mv_y: 0 };
    let meta = partitioned_b_meta(mb_type as u32)?;
    let mv_for = |usage: BListUse| -> BPartitionMv {
        let (mv_l0, mv_l1) = match usage {
            BListUse::L0 => (Some(zero), None),
            BListUse::L1 => (None, Some(zero)),
            BListUse::Bi => (Some(zero), Some(zero)),
        };
        BPartitionMv { mv_l0, mv_l1 }
    };
    Some(BMbDecision::Partitioned {
        mb_type,
        parts: [mv_for(meta.part0), mv_for(meta.part1)],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_match_x264_medium_buckets() {
        // §6E-A6.1 distribution gate: over a large sample the bucket
        // breakdown should approximate the x264-medium reference
        // (50/35/6/5/4 ratio). Tolerate ±3pp wiggle room over 10000
        // samples — hash should be uniform enough.
        //
        // Hold `B_FORCE_MODE_ENV_LOCK` for the duration so parallel
        // tests setting `PHASM_B_FORCE_MODE` (force-mode round-trip
        // suite) can't race and corrupt this test's mb_decision_b
        // calls. Without the lock, an env-var-setting test running
        // concurrently would make some samples return Partitioned /
        // L0_16x16 / etc instead of the bucket distribution.
        let _lock = B_FORCE_MODE_ENV_LOCK.lock().expect("lock not poisoned");
        // SAFETY: serialized via lock; var must be unset for hash-
        // based path.
        unsafe { std::env::remove_var("PHASM_B_FORCE_MODE"); }
        let grid = EncoderMvGrid::new(2, 2);
        let mut counts = [0u32; 5]; // [skip, direct, l0, l1, bi]
        let n = 10_000u32;
        for mb_addr in 0..n {
            let d = mb_decision_b(&grid, 0, 0, /* frame_num */ 0, mb_addr);
            let idx = match d {
                BMbDecision::Skip => 0,
                BMbDecision::Direct16x16 => 1,
                BMbDecision::L0_16x16 { .. } => 2,
                BMbDecision::L1_16x16 { .. } => 3,
                BMbDecision::Bi_16x16 { .. } => 4,
                _ => panic!("unexpected variant from §6E-A6.1 stub"),
            };
            counts[idx] += 1;
        }
        let pct = |c: u32| (c as f32 / n as f32) * 100.0;
        let (skip, direct, l0, l1, bi) =
            (pct(counts[0]), pct(counts[1]), pct(counts[2]), pct(counts[3]), pct(counts[4]));
        eprintln!(
            "§6E-A6.1 mode mix: skip={skip:.1}% direct={direct:.1}% \
             L0={l0:.1}% L1={l1:.1}% Bi={bi:.1}%"
        );
        // Targets: 50 / 35 / 6 / 5 / 4. ±3pp tolerance.
        assert!((skip - 50.0).abs() < 3.0, "skip {skip:.1}%");
        assert!((direct - 35.0).abs() < 3.0, "direct {direct:.1}%");
        assert!((l0 - 6.0).abs() < 3.0, "L0 {l0:.1}%");
        assert!((l1 - 5.0).abs() < 3.0, "L1 {l1:.1}%");
        assert!((bi - 4.0).abs() < 3.0, "Bi {bi:.1}%");
    }

    #[test]
    fn deterministic_output_for_same_input() {
        // Critical for round-trip correctness: encoder + walker
        // both call mb_decision_b at the same (frame_num, mb_addr)
        // and must agree on the result.
        let grid = EncoderMvGrid::new(2, 2);
        for mb_addr in 0..256 {
            let a = mb_decision_b(&grid, 0, 0, 7, mb_addr);
            let b = mb_decision_b(&grid, 0, 0, 7, mb_addr);
            // BMbDecision doesn't impl PartialEq (some variants carry
            // MV data), but the discriminant must match.
            let disc = |d: &BMbDecision| std::mem::discriminant(d);
            assert_eq!(disc(&a), disc(&b),
                "non-deterministic mb_decision at mb={mb_addr}");
        }
    }
}
