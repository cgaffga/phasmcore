// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 6F.2(j) — cascade-safe MVD subset analysis.
//!
//! Computes which MVD-sign positions in cover_p1 can be flipped
//! by the STC plan WITHOUT shifting any downstream MVD across
//! zero (= no cover-shape change between Pass 1 and Pass 3) AND
//! without having any other safe flip shift this position's own
//! magnitude (= magnitude invariant under all selected flips).
//!
//! ## Why
//!
//! Residual-only stego routing concentrates 100% of bypass-bin
//! modifications in a single domain. A steganalysis tool comparing
//! per-domain bypass-bin entropy vs a clean reference sees pristine
//! MVD bins next to perturbed residual bins — that asymmetry is
//! itself a fingerprint, irrespective of per-coefficient
//! detectability. Real H.264 encoders distribute bypass-bin output
//! across all four domains, so a clip with ZERO MVD modifications
//! is suspicious.
//!
//! Restoring some MVD injection (even at modest 13–15% of
//! cover_p1.mvd) breaks the single-domain anomaly. The capacity
//! gain is small; the per-domain modification spread is the
//! load-bearing motivation.
//!
//! ## Criterion C (chicken-and-egg-free)
//!
//! Greedy raster-order selection. At step i, position P_i is
//! added to the safe set IFF:
//!
//! - **Predicate (a)**: no previously-selected P_j has its flip
//!   touching mv_grid cells that P_i's predictor reads. This
//!   guarantees `m_P_i` is invariant under all selected flips,
//!   so the encoder's planning view of `|m_P_i|` equals the
//!   walker's post-cascade view of `|m_P_i|`.
//!
//! - **Predicate (b)**: for every position Q that's later in
//!   raster order AND whose predictor reads any cell P_i would
//!   touch on flip, the cumulative shift bound at Q (= Σ over
//!   already-selected P_k that affect Q of `|2·m_P_k|`, plus the
//!   new contribution from P_i) is strictly less than `|m_Q|`.
//!   This guarantees Q stays nonzero (no cover-shape change).
//!
//! Both predicates depend ONLY on:
//! - position structure (frame_idx, mb_addr, partition, axis) —
//!   sign-flip invariant.
//! - `|m_P|` magnitudes — sign-flip invariant.
//!
//! Walker on Pass 3 bytes captures the same `(position, |m_P|)`
//! tuples (the §6F.2(j).1 magnitude-capture parity gate
//! confirms byte-identity on real-world content). Walker runs
//! identical greedy → identical safe set. No iteration.
//!
//! ## MV-grid propagation pattern
//!
//! Per H.264 spec § 8.4.1.3 (median-of-A/B/C predictor):
//! - A neighbour: cell (tl_bx-1, tl_by) — strictly to the LEFT
//!   of the partition's top-left.
//! - B neighbour: cell (tl_bx, tl_by-1) — strictly ABOVE.
//! - C neighbour: cell (tl_bx + part_w_4x4, tl_by-1) — top-right.
//! - D fallback when C unavailable: (tl_bx-1, tl_by-1).
//!
//! For analysis, we use a CONSERVATIVE MB-level model: an MB
//! propagates to MBs to its right, below, bottom-left, and
//! bottom-right. This may classify a few positions as unsafe
//! that are actually safe (e.g., when the median actually picks
//! a neighbour OTHER than P) but never classifies an unsafe
//! position as safe. Conservative under-approximation is sound:
//! decoder agrees with encoder.
//!
//! ## Output
//!
//! `analyze_safe_mvd_subset(meta, mb_w, mb_h)` returns a
//! `Vec<bool>` aligned with `meta`. `true` at index i means
//! `meta[i]` is safe to flip under the criterion above.

use super::encoder_hook::MvdPositionMeta;
use super::{Axis, PositionKey, SyntaxPath};

/// §6E-A6.0 — hard cap on per-GOP MVD position count.
///
/// Sized for 1080p × 30-frame IBPBP under realistic motion: ~50k
/// positions today (§6E-A4(c)-lite ships zero MVDs from B-frames),
/// projected ~150-300k under §6E-A6.x partitioned-B with realistic
/// content. The 200,000 cap leaves headroom while still catching
/// pathological mode-decision configurations (e.g. all-MBs-emit-
/// `B_Bi_4x4` would balloon to ~2.6M positions per frame and blow
/// per-GOP working-set memory budget).
///
/// The cap is *advisory* in §6E-A4(c)-lite (today's shipping path
/// produces well under 200k positions), but becomes a hard guard
/// rail once §6E-A6.1+ widens the mode set. See
/// `docs/design/video/h264/encoder-algorithms/6E-A6-bslice-partitions.md`
/// open question § 5 (Streaming-Viterbi K window — RESOLVED) for
/// the rationale.
pub const MAX_MVD_POSITIONS_PER_GOP: usize = 200_000;

/// §6E-A6.0 — guard rail check. Returns `Err` if `mvd_count` exceeds
/// the per-GOP working-set budget. Callers (orchestrator, decoder)
/// invoke this before allocating per-position metadata vectors so a
/// pathological GOP fails cleanly with a documented error rather
/// than silently OOM'ing on long-form mobile clips.
///
/// `gop_label` is included in the error string so multi-GOP
/// orchestration can pinpoint the offending GOP.
pub fn check_mvd_budget(
    mvd_count: usize,
    gop_label: &str,
) -> Result<(), crate::stego::error::StegoError> {
    if mvd_count > MAX_MVD_POSITIONS_PER_GOP {
        return Err(crate::stego::error::StegoError::InvalidVideo(format!(
            "MVD position count {mvd_count} exceeds per-GOP streaming \
             budget {MAX_MVD_POSITIONS_PER_GOP} at {gop_label}: \
             encoder mode-decision is producing pathologically fine \
             partitioning (see §6E-A6.0 working-set guard rail)",
        )));
    }
    Ok(())
}

/// Phase 6F.2(j) — analyze which MVD positions are cascade-safe
/// to flip.
///
/// Inputs:
/// - `meta`: per-MVD-position metadata, in cover-emission order
///   (= raster MB scan order, partition order within MB). Element
///   i corresponds to `cover.mvd_sign_bypass.positions[i]`.
/// - `mb_w`, `mb_h`: frame dimensions in macroblocks.
///
/// Output: `Vec<bool>` aligned with `meta`. `true` ⇒ position is
/// in the cascade-safe subset (criterion C).
///
/// Determinism: pure function of inputs, no hidden state, no I/O.
/// Walker calls with identical `meta` → identical `Vec<bool>`.
pub fn analyze_safe_mvd_subset(
    meta: &[MvdPositionMeta],
    mb_w: u32,
    mb_h: u32,
) -> Vec<bool> {
    let n = meta.len();
    let mut safe = vec![false; n];
    if n == 0 || mb_w == 0 || mb_h == 0 {
        return safe;
    }

    // Per-position cumulative-shift accumulator.
    let mut shift_bound: Vec<u32> = vec![0; n];

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| {
        (meta[i].frame_idx, meta[i].mb_addr, meta[i].partition, meta[i].axis)
    });

    // STEGO.A.12.perf — O(n²) → O(n × k) optimization.
    //
    // The two inner loops (predicates a + b) only need to examine
    // positions at MB-level NEIGHBOURS of P (mb_propagates is a
    // 4-cell stencil: right/below/bottom-left/bottom-right). Index
    // positions by (frame_idx, mb_addr) so each predicate is O(k)
    // lookups into ≤ 4 neighbour MBs instead of O(n) scans over
    // every position.
    //
    // At 1024×576 × 150f with 271k MVD positions, the baseline
    // O(n²) took 62 seconds; this O(n × k) variant lands in
    // sub-second territory.
    use std::collections::HashMap;
    let mut mb_index: HashMap<(u32, u32), Vec<usize>> =
        HashMap::with_capacity(n / 8);
    for (i, p) in meta.iter().enumerate() {
        mb_index.entry((p.frame_idx, p.mb_addr)).or_default().push(i);
    }

    // Rank lookup: order_rank[i] = position of i in `order`. Used
    // for the "earlier in raster than P" check in predicate (a).
    let mut order_rank: Vec<u32> = vec![0; n];
    for (rank, &i) in order.iter().enumerate() {
        order_rank[i] = rank as u32;
    }

    // Helper closures to enumerate the 4 propagation-source MBs (for
    // predicate a) and 4 propagation-target MBs (for predicate b).
    // Returns (mb_x, mb_y, mb_addr) for in-bounds MBs only.
    let source_mbs = |p_mx: u32, p_my: u32| -> [Option<u32>; 4] {
        // Sources read FROM these neighbours when computing P's
        // predictor. Equivalently: these neighbours PROPAGATE TO P.
        //   left: (p_mx-1, p_my)
        //   above: (p_mx, p_my-1)
        //   above-left D-fallback: (p_mx-1, p_my-1)
        //   above-right: (p_mx+1, p_my-1)
        let coords = [
            (p_mx.checked_sub(1), Some(p_my)),
            (Some(p_mx), p_my.checked_sub(1)),
            (p_mx.checked_sub(1), p_my.checked_sub(1)),
            (Some(p_mx + 1), p_my.checked_sub(1)),
        ];
        let mut out = [None; 4];
        for (slot, (xo, yo)) in out.iter_mut().zip(coords.iter()) {
            if let (Some(x), Some(y)) = (*xo, *yo) {
                if x < mb_w && y < mb_h {
                    *slot = Some(y * mb_w + x);
                }
            }
        }
        out
    };
    let target_mbs = |p_mx: u32, p_my: u32| -> [Option<u32>; 4] {
        // P propagates TO these neighbours when they read predictors.
        //   right: (p_mx+1, p_my)
        //   below: (p_mx, p_my+1)
        //   below-left: (p_mx-1, p_my+1)
        //   below-right: (p_mx+1, p_my+1)
        let coords = [
            (Some(p_mx + 1), Some(p_my)),
            (Some(p_mx), Some(p_my + 1)),
            (p_mx.checked_sub(1), Some(p_my + 1)),
            (Some(p_mx + 1), Some(p_my + 1)),
        ];
        let mut out = [None; 4];
        for (slot, (xo, yo)) in out.iter_mut().zip(coords.iter()) {
            if let (Some(x), Some(y)) = (*xo, *yo) {
                if x < mb_w && y < mb_h {
                    *slot = Some(y * mb_w + x);
                }
            }
        }
        out
    };

    for &i in &order {
        let p = &meta[i];
        let p_mx = p.mb_addr % mb_w;
        let p_my = p.mb_addr / mb_w;
        let i_rank = order_rank[i];

        // Predicate (a): no EARLIER safe position at a propagation-
        // source MB. Source MB = any MB that propagates TO P's MB,
        // i.e. one of the 4 source neighbours.
        let mut pred_a = true;
        'a: for src_addr in source_mbs(p_mx, p_my).iter().flatten() {
            if let Some(positions) = mb_index.get(&(p.frame_idx, *src_addr)) {
                for &j in positions {
                    if safe[j] && order_rank[j] < i_rank {
                        pred_a = false;
                        break 'a;
                    }
                }
            }
        }
        if !pred_a { continue; }

        // Predicate (b): for every Q at a propagation-TARGET MB of P,
        // the cumulative shift bound + this flip's 2|m_P| must stay
        // < |m_Q|. Target MBs are by construction strictly later than
        // P's MB in raster scan, so we don't need to re-check raster
        // ordering at the MB level.
        let two_m_p = 2u32.saturating_mul(p.magnitude);
        let mut pred_b = true;
        let mut tentative_updates: Vec<usize> = Vec::new();

        'b: for tgt_addr in target_mbs(p_mx, p_my).iter().flatten() {
            if let Some(positions) = mb_index.get(&(p.frame_idx, *tgt_addr)) {
                for &j in positions {
                    let q = &meta[j];
                    let new_bound = shift_bound[j].saturating_add(two_m_p);
                    if new_bound >= q.magnitude {
                        pred_b = false;
                        break 'b;
                    }
                    tentative_updates.push(j);
                }
            }
        }

        if pred_b {
            safe[i] = true;
            for j in tentative_updates {
                shift_bound[j] = shift_bound[j].saturating_add(two_m_p);
            }
        }
    }

    safe
}

/// Returns `true` iff a flip in MB (sx, sy) can affect any
/// predictor read by an MB at (qx, qy). Per H.264 spec § 8.4.1.3,
/// MB (qx, qy)'s partitions' predictors read from neighbours at
/// positions (qx-1, qy), (qx, qy-1), (qx+1, qy-1), and via
/// D-fallback (qx-1, qy-1). Equivalently:
///
/// - (sx, sy) propagates to (sx+1, sy)        — right neighbour
/// - (sx, sy) propagates to (sx, sy+1)        — below
/// - (sx, sy) propagates to (sx-1, sy+1)      — bottom-left (its
///   top-right reads (sx, sy))
/// - (sx, sy) propagates to (sx+1, sy+1)      — bottom-right (its
///   top-left D-fallback reads (sx, sy))
///
/// Conservative: an MB-level affirmative may overcount within-MB
/// granularity, but never under-counts (a true propagation at
/// 4×4-cell granularity always implies MB-level propagation).
#[inline]
pub fn mb_propagates(sx: u32, sy: u32, qx: u32, qy: u32) -> bool {
    // Right: (sx+1, sy)
    if qx == sx.saturating_add(1) && qy == sy { return true; }
    // Below: (sx, sy+1)
    if qx == sx && qy == sy.saturating_add(1) { return true; }
    // Bottom-left: (sx-1, sy+1) — only when sx > 0
    if sx > 0 && qx == sx - 1 && qy == sy.saturating_add(1) { return true; }
    // Bottom-right: (sx+1, sy+1)
    if qx == sx.saturating_add(1) && qy == sy.saturating_add(1) { return true; }
    false
}

/// Strict raster ordering: returns `true` iff (qx, qy) is later
/// than (sx, sy) in MB raster scan order.
#[inline]
fn mb_strictly_later(sx: u32, sy: u32, qx: u32, qy: u32) -> bool {
    qy > sy || (qy == sy && qx > sx)
}

/// §6E-A5(d).2 — map a sign-aligned safe mask to a suffix-aligned
/// safe mask.
///
/// `msb_positions` is `cover.mvd_sign_bypass.positions` (one entry
/// per logged MVD-sign bypass bin). `safe_msb` is the output of
/// [`analyze_safe_mvd_subset`], aligned 1:1 with `msb_positions`.
/// `msl_positions` is `cover.mvd_suffix_lsb.positions`, a subset
/// of MVD slots — only those with `|MVD| ≥ 9` (which emit a UEG3
/// suffix-LSB bin).
///
/// Returns `Vec<bool>` aligned with `msl_positions`. `true` at
/// index `j` means the MVD slot whose suffix-LSB lives at
/// `msl_positions[j]` was marked cascade-safe by `analyze_safe_mvd_subset`.
///
/// Implementation: build a `(frame_idx, mb_addr, list, partition,
/// axis) → bool` lookup table from sign positions + safe_msb;
/// look up each suffix position's tuple. Sign and suffix bins of
/// the same MVD slot share these 5 fields and differ only in
/// `BinKind`, so the lookup is exact.
///
/// Slots whose `SyntaxPath` isn't `Mvd` (would indicate a logging
/// bug) are conservatively marked unsafe — the caller drops them.
pub fn derive_msl_safe_from_msb(
    msb_positions: &[PositionKey],
    safe_msb: &[bool],
    msl_positions: &[PositionKey],
) -> Vec<bool> {
    type SlotKey = (u32, u32, u8, u8, Axis);
    fn slot_key(k: PositionKey) -> Option<SlotKey> {
        match k.syntax_path() {
            SyntaxPath::Mvd { list, partition, axis, .. } =>
                Some((k.frame_idx(), k.mb_addr(), list, partition, axis)),
            _ => None,
        }
    }
    let n = msb_positions.len().min(safe_msb.len());
    let mut map: std::collections::HashMap<SlotKey, bool> =
        std::collections::HashMap::with_capacity(n);
    for i in 0..n {
        if let Some(key) = slot_key(msb_positions[i]) {
            map.insert(key, safe_msb[i]);
        }
    }
    msl_positions
        .iter()
        .map(|&k| {
            slot_key(k)
                .and_then(|sk| map.get(&sk).copied())
                .unwrap_or(false)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(frame_idx: u32, mb_addr: u32, partition: u8, axis: u8, magnitude: u32) -> MvdPositionMeta {
        MvdPositionMeta { frame_idx, mb_addr, partition, axis, magnitude }
    }

    // ─── §6E-A6.0 working-set budget guard ─────────────────────

    #[test]
    fn check_mvd_budget_passes_at_cap() {
        // Exactly at the cap is OK — the cap is `> MAX`, not `>=`.
        assert!(check_mvd_budget(MAX_MVD_POSITIONS_PER_GOP, "test").is_ok());
        assert!(check_mvd_budget(0, "test").is_ok());
        assert!(check_mvd_budget(MAX_MVD_POSITIONS_PER_GOP - 1, "test").is_ok());
    }

    #[test]
    fn check_mvd_budget_rejects_above_cap() {
        let r = check_mvd_budget(MAX_MVD_POSITIONS_PER_GOP + 1, "gop[5]");
        assert!(r.is_err());
        // Error message should include the count, the budget, and
        // the gop_label so a multi-GOP orchestrator can localize.
        let msg = format!("{:?}", r.unwrap_err());
        assert!(msg.contains("gop[5]"), "missing gop label: {msg}");
        assert!(msg.contains(&MAX_MVD_POSITIONS_PER_GOP.to_string()),
            "missing cap value: {msg}");
    }

    #[test]
    fn check_mvd_budget_const_is_nonzero_and_realistic() {
        // Sanity: 200k positions × 44 bytes/MvdPositionMeta+plan
        // = ~8.8 MB worst-case per GOP, well within the 250-400 MB
        // total Phase 6 streaming budget across 2-3 GOPs in flight.
        assert!(MAX_MVD_POSITIONS_PER_GOP >= 50_000);
        assert!(MAX_MVD_POSITIONS_PER_GOP <= 1_000_000);
    }

    #[test]
    fn empty_input_yields_empty_safe_set() {
        let r = analyze_safe_mvd_subset(&[], 4, 3);
        assert!(r.is_empty());
    }

    #[test]
    fn single_position_in_last_mb_is_safe() {
        // Frame is 4×3 MBs; last MB is (3, 2) ⇒ mb_addr = 11.
        let m = vec![meta(0, 11, 0, 0, 1)];
        let r = analyze_safe_mvd_subset(&m, 4, 3);
        assert_eq!(r, vec![true]);
    }

    #[test]
    fn cascade_blocks_when_downstream_magnitude_is_small() {
        // P at MB(0,0) magnitude 5 → 2·5 = 10. Q at MB(1,0)
        // magnitude 3. Flipping P would shift Q's predictor by up to
        // 10, exceeding |m_Q|=3 → P unsafe.
        let m = vec![
            meta(0, 0, 0, 0, 5),  // P at (0,0)
            meta(0, 1, 0, 0, 3),  // Q at (1,0) — propagation target
        ];
        let r = analyze_safe_mvd_subset(&m, 4, 3);
        assert_eq!(r[0], false, "P should be unsafe (cascade exceeds Q's magnitude)");
        // Q is checked after P; P wasn't selected so no constraint
        // on Q. Q has no successors in this 1-MB-deep test (Q is at
        // (1,0); successors at (2,0), (0,1), (1,1), (2,1) — none of
        // those have MVDs in this fixture). So Q is safe.
        assert_eq!(r[1], true);
    }

    #[test]
    fn cascade_safe_upstream_blocks_downstream_via_predicate_a() {
        // P at MB(0,0) magnitude 1 → 2·1 = 2. Q at MB(1,0)
        // magnitude 5. 2 < 5, so P passes predicate (b).
        // P is upstream of Q → P selected.
        // Q would be safe under (b) (it has no propagation
        // targets in this fixture), BUT P (already selected)
        // propagates to Q → Q's predictor is shifted → Q's
        // magnitude not invariant → predicate (a) fails.
        // → r = [true, false].
        let m = vec![
            meta(0, 0, 0, 0, 1),
            meta(0, 1, 0, 0, 5),
        ];
        let r = analyze_safe_mvd_subset(&m, 4, 3);
        assert_eq!(r, vec![true, false]);
    }

    #[test]
    fn non_propagating_pair_both_safe() {
        // P at MB(0,0) magnitude 1; Q at MB(2,2) magnitude 1.
        // (0,0) propagates to (1,0), (0,1), (1,1) — NOT to (2,2).
        // Q is independent of P. Both safe.
        let m = vec![
            meta(0, 0, 0, 0, 1),         // (0,0)
            meta(0, 2 + 2 * 4, 0, 0, 1), // (2,2) ⇒ mb_addr = 10
        ];
        let r = analyze_safe_mvd_subset(&m, 4, 3);
        assert_eq!(r, vec![true, true]);
    }

    #[test]
    fn cumulative_shift_predicate_b_blocks_p0_when_p1_magnitude_low() {
        // P0 at (0,0) mag 1 → 2·1 = 2 shift on downstream MBs.
        // P1 at (0,1) mag 2 — (0,0) propagates to (0,1) below.
        // 2·m_P0 = 2 vs |m_P1| = 2. predicate (b): need
        // shift_bound[P1] < |m_P1|, i.e. 2 < 2 — FALSE
        // (strict). P0 rejected because its flip would push
        // P1's magnitude to zero.
        let m = vec![
            meta(0, 0, 0, 0, 1),
            meta(0, 4, 0, 0, 2), // (0,1) ⇒ mb_addr = 4
        ];
        let r = analyze_safe_mvd_subset(&m, 4, 3);
        assert_eq!(r[0], false, "P0's flip would zero-cross P1");
    }

    #[test]
    fn different_frames_are_independent() {
        // Same MB position but different frame_idx → no
        // cross-frame propagation (each frame has its own
        // mv_grid). Both safe.
        let m = vec![
            meta(0, 0, 0, 0, 1),
            meta(1, 0, 0, 0, 1),
        ];
        let r = analyze_safe_mvd_subset(&m, 4, 3);
        assert_eq!(r, vec![true, true]);
    }

    #[test]
    fn mb_propagates_exhaustive_grid() {
        // Spot-check the four propagation patterns.
        assert!(mb_propagates(1, 1, 2, 1));  // right
        assert!(mb_propagates(1, 1, 1, 2));  // below
        assert!(mb_propagates(1, 1, 0, 2));  // bottom-left
        assert!(mb_propagates(1, 1, 2, 2));  // bottom-right
        // Non-propagation cases.
        assert!(!mb_propagates(1, 1, 1, 1));  // self
        assert!(!mb_propagates(1, 1, 0, 1));  // left (upstream, not down)
        assert!(!mb_propagates(1, 1, 1, 0));  // above
        assert!(!mb_propagates(1, 1, 3, 1));  // 2 right
        assert!(!mb_propagates(1, 1, 0, 0));  // top-left (upstream)
    }

    use super::super::{BinKind, EmbedDomain};

    #[test]
    fn derive_msl_safe_maps_sign_to_suffix_via_position_tuple() {
        // 3 MVD slots at frame 0, MBs (0,0), (0,1), (0,2).
        // All same list/partition/axis — only mb_addr differs.
        // Slot 0 + slot 2 are sign-only (|MVD| < 9 → no suffix).
        // Slot 1 has |MVD| ≥ 9 → emits suffix-LSB.
        let mk_path = |kind: BinKind| SyntaxPath::Mvd {
            list: 0, partition: 0, axis: Axis::X, kind,
        };
        let sign_keys = [
            PositionKey::new(0, 0, EmbedDomain::MvdSignBypass, mk_path(BinKind::Sign)),
            PositionKey::new(0, 4, EmbedDomain::MvdSignBypass, mk_path(BinKind::Sign)),
            PositionKey::new(0, 8, EmbedDomain::MvdSignBypass, mk_path(BinKind::Sign)),
        ];
        let safe_sign = vec![true, false, true];
        let suffix_keys = [
            // Suffix only for slot 1 (mb_addr=4).
            PositionKey::new(0, 4, EmbedDomain::MvdSuffixLsb, mk_path(BinKind::SuffixLsb)),
        ];
        let safe_suffix = derive_msl_safe_from_msb(&sign_keys, &safe_sign, &suffix_keys);
        assert_eq!(safe_suffix.len(), 1);
        assert_eq!(safe_suffix[0], false, "suffix slot 1 should match sign safe[1] = false");
    }

    #[test]
    fn derive_msl_safe_handles_orthogonal_axes() {
        // One MVD with X + Y components in the same partition →
        // two sign positions (axis=X, axis=Y), one suffix per
        // axis. Y is unsafe, X is safe.
        let mk_path = |axis: Axis, kind: BinKind| SyntaxPath::Mvd {
            list: 0, partition: 0, axis, kind,
        };
        let sign_keys = [
            PositionKey::new(0, 0, EmbedDomain::MvdSignBypass, mk_path(Axis::X, BinKind::Sign)),
            PositionKey::new(0, 0, EmbedDomain::MvdSignBypass, mk_path(Axis::Y, BinKind::Sign)),
        ];
        let safe_sign = vec![true, false];
        let suffix_keys = [
            PositionKey::new(0, 0, EmbedDomain::MvdSuffixLsb, mk_path(Axis::Y, BinKind::SuffixLsb)),
            PositionKey::new(0, 0, EmbedDomain::MvdSuffixLsb, mk_path(Axis::X, BinKind::SuffixLsb)),
        ];
        let safe_suffix = derive_msl_safe_from_msb(&sign_keys, &safe_sign, &suffix_keys);
        assert_eq!(safe_suffix, vec![false, true]);
    }

    #[test]
    fn derive_msl_safe_returns_empty_for_empty_msl() {
        let safe = derive_msl_safe_from_msb(&[], &[], &[]);
        assert!(safe.is_empty());
    }
}
