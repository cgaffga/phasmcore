// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! §long-form-stego Phase 6.2 — per-GOP plan builder for the
//! interleaved Phase B orchestrator.
//!
//! Phase 5's streaming orchestrator runs each domain's
//! streaming-Viterbi to completion before slicing the resulting
//! O(n) `DomainPlan` per GOP and firing Pass 3. At 15-min 1080p the
//! resident plan size for the active domains hits ~3.5 GB, which
//! pushes mobile peak RSS over the OOM cliff.
//!
//! The Phase 6 architecture (Choice β, locked 2026-04-30):
//! interleaved Phase B emission with **per-GOP plan completion
//! firing**. Plan bits flow directly from
//! `StreamingViterbiPhaseB::step()` calls into per-GOP per-domain
//! buffers, fire Pass 3 as GOPs complete, drop their plan buffers.
//!
//! `PerGopPlanBuilder` is the routing data structure that owns the
//! per-GOP per-domain buffers and tracks completion. The Phase 6.3
//! orchestrator drives 4× `StreamingViterbiPhaseB` in round-robin
//! lockstep, threads each emission through `accept_emission`, and
//! polls `take_ready_gops` after every step to fire Pass 3
//! incrementally.
//!
//! ## Memory bound
//!
//! `O(num_gops × 4 × usize)` cumulative-counts plus the active
//! per-GOP buffers. Since `take_ready_gops` drops fired GOPs
//! immediately, the resident buffer count tracks how many GOPs are
//! "in flight" — typically ~few at a time when emissions arrive in
//! reverse order from the lockstep drivers. Per 1080p GOP at
//! ~32k positions/domain × 4 domains × 1 byte = ~128 KB per GOP;
//! a few GOPs in flight = a handful of MB resident.

use std::array;

/// One ready-to-fire GOP's plan, returned by `take_ready_gops`.
///
/// `plans[d]` is the per-domain plan slice for domain `d`,
/// length `cum_counts[d][gop_idx + 1] - cum_counts[d][gop_idx]`
/// (or empty for inactive domains).
#[derive(Debug, Clone)]
pub struct ReadyGop {
    pub gop_idx: usize,
    pub plans: [Vec<u8>; 4],
}

/// Per-GOP plan buffer + completion tracker for the Phase 6
/// interleaved orchestrator.
///
/// **Allocation policy**: per-GOP per-domain buffers are
/// **allocated lazily** on the first emission for that
/// `(gop, domain)` pair. Buffers for inactive domains and
/// zero-length GOP slots are never allocated. After
/// `take_ready_gops` fires a GOP, its 4 buffer slots are taken
/// out and the builder holds `None`s in their place.
///
/// At peak, the resident buffer set is therefore bounded by the
/// number of GOPs currently in flight from the lockstep emission
/// loop — typically a handful, set by how synchronized the
/// per-domain drivers are. For 15-min 1080p (~90 GOPs), this
/// keeps the plan-side memory near ~tens of MB instead of the
/// ~3.5 GB v1 holds for the full plans.
pub struct PerGopPlanBuilder {
    /// `cum_counts[d][g]` = positions for domain `d` in GOPs
    /// `[0..g)`. Length `num_gops + 1`. Per-GOP per-domain capacity
    /// derives from `cum_counts[d][g+1] - cum_counts[d][g]`.
    cum_counts: [Vec<usize>; 4],
    /// `gop_buffers[g][d]` = plan bits for GOP `g`, domain `d`.
    /// `None` until first emission allocates it (lazy). Taken out
    /// by `take_ready_gops` once the GOP fires.
    gop_buffers: Vec<[Option<Vec<u8>>; 4]>,
    /// `gop_filled[g][d]` = bytes-written counter. Domain `d`'s
    /// contribution to GOP `g` is complete when this equals
    /// `cum_counts[d][g+1] - cum_counts[d][g]` (or 0 for inactive
    /// / empty-slot domains, in which case it's pre-marked done).
    gop_filled: Vec<[usize; 4]>,
    /// `gop_domain_done[g][d]` cached completion flag. Pre-set to
    /// `true` for inactive or zero-length domain entries so
    /// `take_ready_gops` can fire the GOP without waiting on those.
    gop_domain_done: Vec<[bool; 4]>,
    /// Whether GOP `g` has been fired (returned via
    /// `take_ready_gops`).
    gop_fired: Vec<bool>,
    num_gops: usize,
    /// Pass-through accumulators for end-of-encode reporting.
    total_modifications: usize,
    total_cost: f64,
}

impl PerGopPlanBuilder {
    /// Construct from per-GOP per-domain counts and an "active"
    /// mask. `active_domains[d] == false` means no
    /// `accept_emission` calls will arrive for domain `d`; the
    /// builder pre-marks that domain done for every GOP and yields
    /// an empty `plans[d]` slice in `ReadyGop`.
    ///
    /// Buffers are NOT allocated here — only the per-domain
    /// cumulative-count arrays + completion flags. The first
    /// `accept_emission` call for a given `(gop, domain)` allocates
    /// the buffer lazily, sized to `cum_counts[d][gop+1] -
    /// cum_counts[d][gop]`.
    pub fn new(
        per_gop_counts: &[[usize; 4]],
        active_domains: [bool; 4],
    ) -> Self {
        let num_gops = per_gop_counts.len();
        let cum_counts: [Vec<usize>; 4] = array::from_fn(|d| {
            let mut v = Vec::with_capacity(num_gops + 1);
            v.push(0);
            for row in per_gop_counts.iter() {
                v.push(*v.last().unwrap() + row[d]);
            }
            v
        });
        // Lazy alloc: every slot starts as None. Filled on demand
        // by accept_emission (or stays None for inactive / zero-
        // length GOP-domain pairs, which yield empty Vec at
        // take_ready_gops time).
        let gop_buffers: Vec<[Option<Vec<u8>>; 4]> =
            (0..num_gops).map(|_| [None, None, None, None]).collect();
        let gop_filled = vec![[0usize; 4]; num_gops];
        let gop_domain_done: Vec<[bool; 4]> = (0..num_gops)
            .map(|g| {
                array::from_fn(|d| {
                    !active_domains[d] || per_gop_counts[g][d] == 0
                })
            })
            .collect();
        let gop_fired = vec![false; num_gops];
        Self {
            cum_counts,
            gop_buffers,
            gop_filled,
            gop_domain_done,
            gop_fired,
            num_gops,
            total_modifications: 0,
            total_cost: 0.0,
        }
    }

    /// Total positions for domain `d` across the whole clip.
    pub fn total_positions(&self, domain_idx: usize) -> usize {
        *self.cum_counts[domain_idx].last().unwrap_or(&0)
    }

    /// Number of GOPs.
    pub fn num_gops(&self) -> usize {
        self.num_gops
    }

    /// Accumulated modification count from
    /// `PhaseBSegmentEmission::num_modifications` summing.
    pub fn total_modifications(&self) -> usize {
        self.total_modifications
    }

    /// Accumulated cost from per-domain `total_cost()`.
    pub fn total_cost(&self) -> f64 {
        self.total_cost
    }

    /// Pass-through accumulator: add to running modification count.
    pub fn add_modifications(&mut self, n: usize) {
        self.total_modifications += n;
    }

    /// Pass-through accumulator: add to running total cost.
    pub fn add_cost(&mut self, c: f64) {
        self.total_cost += c;
    }

    /// True once every GOP has been fired by `take_ready_gops`.
    pub fn all_fired(&self) -> bool {
        self.gop_fired.iter().all(|&f| f)
    }

    /// Highest GOP index where domain `domain_idx` still has pending
    /// (not-yet-emitted) bits. Returns `None` if every GOP for that
    /// domain has reported completion.
    ///
    /// Phase 6.3's interleaved orchestrator uses this to keep all 4
    /// `StreamingViterbiPhaseB` drivers in approximate GOP-sync —
    /// it always advances the driver whose `highest_pending_gop` is
    /// largest, which bounds the in-flight per-GOP buffer set to a
    /// handful of GOPs at a time.
    pub fn highest_pending_gop(&self, domain_idx: usize) -> Option<usize> {
        if domain_idx >= 4 {
            return None;
        }
        (0..self.num_gops).rev().find(|&g| !self.gop_domain_done[g][domain_idx])
    }

    /// Accept one Phase B segment emission. Splits `stego_bits`
    /// across the GOPs the position range `[j_start..j_start +
    /// stego_bits.len())` spans, copies into the appropriate per-GOP
    /// per-domain buffers, updates completion counters.
    ///
    /// Empty `stego_bits` is a no-op. Returns `Err` on out-of-range
    /// `domain_idx`, position-range overrun, or attempt to write
    /// into an already-fired GOP.
    pub fn accept_emission(
        &mut self,
        domain_idx: usize,
        j_start: usize,
        stego_bits: &[u8],
    ) -> Result<(), &'static str> {
        if domain_idx >= 4 {
            return Err("domain_idx out of range");
        }
        if stego_bits.is_empty() {
            return Ok(());
        }
        let j_end = j_start + stego_bits.len();
        let total = self.total_positions(domain_idx);
        if j_end > total {
            return Err("emission overruns cumulative count");
        }

        let cum = &self.cum_counts[domain_idx];
        // gop_start = largest g such that cum[g] <= j_start.
        let gop_start = cum.partition_point(|&c| c <= j_start).saturating_sub(1);
        // gop_end = smallest g such that cum[g] >= j_end.
        let gop_end = cum.partition_point(|&c| c < j_end);

        let mut bits_offset = 0usize;
        for g in gop_start..gop_end {
            let gop_lo = cum[g];
            let gop_hi = cum[g + 1];
            let copy_lo = j_start.max(gop_lo);
            let copy_hi = j_end.min(gop_hi);
            if copy_hi <= copy_lo {
                continue;
            }
            let copy_len = copy_hi - copy_lo;
            let target_lo = copy_lo - gop_lo;
            let target_hi = target_lo + copy_len;

            // Lazy alloc: create the buffer the first time we
            // write to (g, domain_idx). Once fired, the slot is
            // None AND gop_fired[g] is true → that's how we
            // distinguish "lazy unallocated" from "already fired".
            if self.gop_fired[g] {
                return Err("GOP buffer already fired");
            }
            let expected = gop_hi - gop_lo;
            let buf = self.gop_buffers[g][domain_idx]
                .get_or_insert_with(|| vec![0u8; expected]);
            buf[target_lo..target_hi]
                .copy_from_slice(&stego_bits[bits_offset..bits_offset + copy_len]);
            self.gop_filled[g][domain_idx] += copy_len;

            if self.gop_filled[g][domain_idx] >= expected {
                self.gop_domain_done[g][domain_idx] = true;
            }
            bits_offset += copy_len;
        }
        if bits_offset != stego_bits.len() {
            return Err("emission span calc mismatch");
        }
        Ok(())
    }

    /// Take all GOPs ready to fire (all 4 domains complete) and
    /// not yet fired. Returns them in ascending `gop_idx` order.
    /// After this call, the returned GOPs' buffers are removed
    /// from the builder so the caller's Pass 3 driver owns the
    /// memory.
    pub fn take_ready_gops(&mut self) -> Vec<ReadyGop> {
        let mut out = Vec::new();
        for g in 0..self.num_gops {
            if self.gop_fired[g] {
                continue;
            }
            if !self.gop_domain_done[g].iter().all(|&b| b) {
                continue;
            }
            let plans: [Vec<u8>; 4] = array::from_fn(|d| {
                self.gop_buffers[g][d].take().unwrap_or_default()
            });
            self.gop_fired[g] = true;
            out.push(ReadyGop { gop_idx: g, plans });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single-domain, single-GOP: a single emission completes the
    /// GOP and `take_ready_gops` returns it.
    #[test]
    fn single_domain_single_gop_fires_immediately() {
        // 1 GOP, domain 0 has 8 positions, domains 1-3 inactive.
        let counts = vec![[8, 0, 0, 0]];
        let mut b = PerGopPlanBuilder::new(&counts, [true, false, false, false]);
        assert_eq!(b.num_gops(), 1);
        assert_eq!(b.total_positions(0), 8);

        // No GOPs ready before any emission.
        assert!(b.take_ready_gops().is_empty());

        let bits: Vec<u8> = (0..8u8).map(|i| i & 1).collect();
        b.accept_emission(0, 0, &bits).unwrap();

        let ready = b.take_ready_gops();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].gop_idx, 0);
        assert_eq!(ready[0].plans[0], bits);
        assert!(ready[0].plans[1].is_empty());
        assert!(ready[0].plans[2].is_empty());
        assert!(ready[0].plans[3].is_empty());
        assert!(b.all_fired());
    }

    /// Cross-GOP segment: a single emission spanning two GOPs
    /// routes correctly into both per-GOP buffers.
    #[test]
    fn cross_gop_segment_routes_to_both_gops() {
        // 2 GOPs, domain 0: [5, 5] = 10 total. Other domains inactive.
        let counts = vec![[5, 0, 0, 0], [5, 0, 0, 0]];
        let mut b =
            PerGopPlanBuilder::new(&counts, [true, false, false, false]);

        // Emission spans j=[3..8) — last 2 of GOP 0 + first 3 of GOP 1.
        let bits: Vec<u8> = vec![1, 0, 1, 1, 0];
        b.accept_emission(0, 3, &bits).unwrap();

        // Neither GOP is ready yet — GOP 0 needs positions [0..3),
        // GOP 1 needs positions [8..10).
        assert!(b.take_ready_gops().is_empty());

        // Emit prefix of GOP 0.
        b.accept_emission(0, 0, &[0, 1, 0]).unwrap();
        let ready = b.take_ready_gops();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].gop_idx, 0);
        assert_eq!(ready[0].plans[0], vec![0, 1, 0, 1, 0]);

        // Emit tail of GOP 1.
        b.accept_emission(0, 8, &[1, 1]).unwrap();
        let ready = b.take_ready_gops();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].gop_idx, 1);
        assert_eq!(ready[0].plans[0], vec![1, 1, 0, 1, 1]);
        assert!(b.all_fired());
    }

    /// Four active domains: GOP doesn't fire until ALL FOUR have
    /// reported completion.
    #[test]
    fn four_domain_gop_waits_for_all_four() {
        let counts = vec![[4, 4, 4, 4]];
        let mut b = PerGopPlanBuilder::new(&counts, [true; 4]);

        b.accept_emission(0, 0, &[0, 1, 0, 1]).unwrap();
        assert!(b.take_ready_gops().is_empty());
        b.accept_emission(1, 0, &[1, 0, 1, 0]).unwrap();
        assert!(b.take_ready_gops().is_empty());
        b.accept_emission(2, 0, &[1, 1, 0, 0]).unwrap();
        assert!(b.take_ready_gops().is_empty());
        b.accept_emission(3, 0, &[0, 0, 1, 1]).unwrap();

        let ready = b.take_ready_gops();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].gop_idx, 0);
        assert_eq!(ready[0].plans[0], vec![0, 1, 0, 1]);
        assert_eq!(ready[0].plans[1], vec![1, 0, 1, 0]);
        assert_eq!(ready[0].plans[2], vec![1, 1, 0, 0]);
        assert_eq!(ready[0].plans[3], vec![0, 0, 1, 1]);
    }

    /// Inactive domains auto-complete and don't block GOP firing.
    #[test]
    fn inactive_domain_auto_completes() {
        let counts = vec![[4, 0, 4, 0]];
        // Domain 0 + 2 active. Domains 1 + 3 inactive (mvd_suffix
        // forced 0 by StealthAllocator).
        let mut b = PerGopPlanBuilder::new(
            &counts,
            [true, false, true, false],
        );
        b.accept_emission(0, 0, &[1, 1, 0, 0]).unwrap();
        // Not ready yet — domain 2 hasn't reported.
        assert!(b.take_ready_gops().is_empty());
        b.accept_emission(2, 0, &[0, 1, 0, 1]).unwrap();
        let ready = b.take_ready_gops();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].plans[0], vec![1, 1, 0, 0]);
        assert!(ready[0].plans[1].is_empty());
        assert_eq!(ready[0].plans[2], vec![0, 1, 0, 1]);
        assert!(ready[0].plans[3].is_empty());
    }

    /// Reverse-segment-order arrival pattern (matches the
    /// streaming-Viterbi Phase B emission order): later GOPs fire
    /// first, earlier GOPs fire last.
    #[test]
    fn reverse_order_gops_fire_in_reverse() {
        let counts = vec![[3, 0, 0, 0], [3, 0, 0, 0], [3, 0, 0, 0]];
        let mut b =
            PerGopPlanBuilder::new(&counts, [true, false, false, false]);

        // Emit GOP 2 first.
        b.accept_emission(0, 6, &[1, 0, 1]).unwrap();
        let ready = b.take_ready_gops();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].gop_idx, 2);
        assert_eq!(ready[0].plans[0], vec![1, 0, 1]);

        // Emit GOP 1.
        b.accept_emission(0, 3, &[0, 1, 0]).unwrap();
        let ready = b.take_ready_gops();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].gop_idx, 1);

        // Emit GOP 0.
        b.accept_emission(0, 0, &[1, 1, 1]).unwrap();
        let ready = b.take_ready_gops();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].gop_idx, 0);
        assert!(b.all_fired());
    }

    /// Multiple emissions for the same domain/GOP accumulate
    /// correctly (e.g., one segment crosses a GOP boundary, leaving
    /// a partial GOP buffer; a later segment completes it).
    #[test]
    fn partial_then_completing_emission_fires_gop() {
        let counts = vec![[5, 0, 0, 0]];
        let mut b =
            PerGopPlanBuilder::new(&counts, [true, false, false, false]);
        b.accept_emission(0, 0, &[1, 0, 1]).unwrap();
        assert!(b.take_ready_gops().is_empty());
        b.accept_emission(0, 3, &[0, 1]).unwrap();
        let ready = b.take_ready_gops();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].plans[0], vec![1, 0, 1, 0, 1]);
    }

    /// Empty emission is a no-op.
    #[test]
    fn empty_emission_is_noop() {
        let counts = vec![[0, 0, 0, 0]];
        let mut b = PerGopPlanBuilder::new(&counts, [true; 4]);
        b.accept_emission(0, 0, &[]).unwrap();
        // GOP 0 has zero positions for every domain → all done from
        // the start.
        let ready = b.take_ready_gops();
        assert_eq!(ready.len(), 1);
        assert!(ready[0].plans.iter().all(|p| p.is_empty()));
    }

    /// Accepting more bits than the cumulative count returns Err.
    #[test]
    fn overrun_returns_err() {
        let counts = vec![[4, 0, 0, 0]];
        let mut b =
            PerGopPlanBuilder::new(&counts, [true, false, false, false]);
        let bits = vec![0u8; 5];
        assert!(b.accept_emission(0, 0, &bits).is_err());
    }

    /// Re-firing an already-fired GOP returns Err.
    #[test]
    fn write_after_fire_returns_err() {
        let counts = vec![[2, 0, 0, 0]];
        let mut b =
            PerGopPlanBuilder::new(&counts, [true, false, false, false]);
        b.accept_emission(0, 0, &[1, 1]).unwrap();
        let _ready = b.take_ready_gops();
        // GOP 0 is fired; another write to its range is an error.
        assert!(b.accept_emission(0, 0, &[0, 0]).is_err());
    }

    /// Pass-through totals accumulate correctly.
    #[test]
    fn pass_through_totals_accumulate() {
        let counts = vec![[2, 0, 0, 0]];
        let mut b =
            PerGopPlanBuilder::new(&counts, [true, false, false, false]);
        b.add_modifications(7);
        b.add_modifications(3);
        b.add_cost(1.5);
        b.add_cost(2.5);
        assert_eq!(b.total_modifications(), 10);
        assert!((b.total_cost() - 4.0).abs() < 1e-9);
    }

    /// Multi-GOP fixture mirroring the streaming-orchestrator's
    /// real shape: 4 GOPs, all 4 domains active, emissions arrive
    /// in reverse-segment order across all 4 domains in
    /// round-robin lockstep. End state: every GOP fired in
    /// ascending order, full plan reconstructed.
    #[test]
    fn realistic_lockstep_pattern() {
        // 4 GOPs, [10, 8, 6, 4] positions per GOP per domain
        // (heterogeneous to catch alignment bugs).
        let counts = vec![
            [10, 10, 10, 10],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [4, 4, 4, 4],
        ];
        let mut b = PerGopPlanBuilder::new(&counts, [true; 4]);
        let total_per_domain = 10 + 8 + 6 + 4;

        // Segment-of-7 reverse-order emissions per domain. With
        // total_per_domain=28, we get 4 segments of size 7.
        let seg_size = 7;
        for d in 0..4 {
            // Reverse-order segments: j_start = 21, 14, 7, 0.
            for seg_idx_rev in (0..4).rev() {
                let j = seg_idx_rev * seg_size;
                let bits: Vec<u8> = (0..seg_size)
                    .map(|i| ((d * 100 + j + i) & 1) as u8)
                    .collect();
                b.accept_emission(d, j, &bits).unwrap();
            }
        }
        // All emissions arrived; drain the ready queue at the end.
        let mut all_ready = Vec::new();
        loop {
            let r = b.take_ready_gops();
            if r.is_empty() {
                break;
            }
            all_ready.extend(r);
        }
        assert_eq!(all_ready.len(), 4);
        // Verify GOPs returned in ascending order.
        for (i, rg) in all_ready.iter().enumerate() {
            assert_eq!(rg.gop_idx, i);
            for d in 0..4 {
                let expected_len = counts[i][d];
                assert_eq!(
                    rg.plans[d].len(),
                    expected_len,
                    "GOP {i} domain {d} length mismatch",
                );
            }
        }
        assert!(b.all_fired());
    }
}
