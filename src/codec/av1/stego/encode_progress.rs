// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Whole-encode progress for the AV1 streaming encode: a 0–1 fraction
//! proportional to the work done, plus the phase and GOP it is on.
//!
//! The two-pass encode's cost is not known up front: Pass 2 is ~5 % of Pass 1
//! while the Pass-1 cache holds every GOP's packets (each is rebuilt by a
//! decode), but every GOP the cache could not hold is natural-encoded again.
//! Packets are small (1–2.4 MB per 1080p GOP), so that takes a very long
//! clip or a tiny budget — yet a fixed split between the passes would then
//! race or sit at 90 % for half the encode. Instead the encode counts work in frame-encode
//! units, re-estimates the total as the cache fills
//! ([`RecomputeForecast`]), and [`WorkProgress`] turns that into a fraction
//! that never moves back.

use crate::stego::gop_cache::GopCacheStats;

/// What the encode is doing when it reports progress.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1EncodePhase {
    /// Pass 1: natural encode of every GOP.
    Encode = 0,
    /// Pass 2: embedding each GOP (rebuilding it from its cached packets
    /// first, or re-encoding any GOP the Pass-1 cache did not hold). A re-planned sweep reports this phase again.
    Embed = 1,
}

/// One progress event of the AV1 streaming encode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Av1EncodeProgress {
    /// Whole-encode fraction: never decreases, 1.0 only when the encode is
    /// done.
    pub fraction: f32,
    pub phase: Av1EncodePhase,
    /// GOPs finished in this phase, 1-based, out of `n_gops`.
    pub gop: u32,
    pub n_gops: u32,
}

/// Cost of embedding one frame in Pass 2 — rebuild (dav1d decode of the
/// cached packets + harvest) + STC + replay — in units of natural-encoding
/// it in Pass 1. Measured 2026-10-24 with every GOP cached (IMG_4138 1080p
/// q100, 2 GOPs of 30, `pull_no_shadow_pass2_cost_ratio`) on a machine
/// under varying load: 0.041–0.047 while Pass 1 still harvested, 0.023–
/// 0.030 since it only counts cover. On cgPhone (iPhone 16 Pro, the iphone5
/// 1080p clip, `ios/Phasm/Debug/Av1EncodeBench.swift`) it was 0.065 — 40.5 s
/// against 627.5 s, MP4 mux included, thermal "fair". The ratio moves with
/// load and device (Pass 2's decode is single-threaded, rav1e is not), so
/// the weight sits above every measurement: too high only slows the bar in
/// Pass 1, too low would park it near the end.
pub(crate) const EMBED_WEIGHT: f64 = 0.07;

/// Monotone fraction of a job whose total is re-estimated while it runs.
///
/// A new estimate keeps the bar where it is and spreads the rest of the bar
/// over the work now expected to remain, so the bar slows down or speeds up
/// but never moves back. Only [`finish`](Self::finish) reports 1.0, so an
/// estimate that turns out low parks the bar just short of the end instead of
/// claiming completion.
pub(crate) struct WorkProgress {
    total: f64,
    done: f64,
    anchor_done: f64,
    anchor_fraction: f64,
    last: f64,
}

/// Highest fraction reported before [`WorkProgress::finish`].
const BEFORE_DONE: f64 = 0.999;

impl WorkProgress {
    pub(crate) fn new(total: f64) -> Self {
        Self {
            total,
            done: 0.0,
            anchor_done: 0.0,
            anchor_fraction: 0.0,
            last: 0.0,
        }
    }

    /// Current estimate of the total work.
    pub(crate) fn total(&self) -> f64 {
        self.total
    }

    /// Work recorded so far.
    pub(crate) fn done(&self) -> f64 {
        self.done
    }

    /// Re-estimate the total work (never below what is already done).
    pub(crate) fn set_total(&mut self, total: f64) {
        self.anchor_done = self.done;
        self.anchor_fraction = self.last;
        self.total = total.max(self.done);
    }

    /// Record `work` more units done; returns the fraction to show.
    pub(crate) fn advance(&mut self, work: f64) -> f32 {
        self.done += work;
        let span = self.total - self.anchor_done;
        let along = if span > 0.0 {
            ((self.done - self.anchor_done) / span).min(1.0)
        } else {
            1.0
        };
        let fraction = self.anchor_fraction + (1.0 - self.anchor_fraction) * along;
        self.last = fraction.min(BEFORE_DONE).max(self.last);
        self.last as f32
    }

    /// The job is done.
    pub(crate) fn finish(&mut self) -> f32 {
        self.last = 1.0;
        1.0
    }
}

/// Predicts how many frames Pass 2 will natural-encode again because the
/// Pass-1 cache cannot hold their GOP, from the cache's fill as Pass 1 runs.
///
/// The cache stores in RAM until its budget is spent, then on disk, then
/// drops entries (Pass 2 recomputes those). After each Pass-1 insert the
/// forecast notes whether that GOP was stored and how big entries are, and
/// predicts that the GOPs still to come fill the remaining room in order.
/// Once one GOP has been dropped, all later ones are assumed dropped too.
/// After Pass 1, [`unstored_frames`](Self::unstored_frames) is exact.
pub(crate) struct RecomputeForecast {
    ram_budget: usize,
    disk_budget: usize,
    gop_frames: Vec<u32>,
    stored: Vec<bool>,
    last: Option<GopCacheStats>,
    ram_entry: usize,
    disk_entry: usize,
}

impl RecomputeForecast {
    /// `gop_frames[g]` is GOP `g`'s frame count.
    pub(crate) fn new(ram_budget: usize, disk_budget: usize, gop_frames: Vec<u32>) -> Self {
        Self {
            ram_budget,
            disk_budget,
            gop_frames,
            stored: Vec::new(),
            last: None,
            ram_entry: 0,
            disk_entry: 0,
        }
    }

    /// Record the next GOP's insert, from the cache stats right after it.
    pub(crate) fn observe(&mut self, stats: GopCacheStats) {
        let before = self.last.unwrap_or(GopCacheStats {
            cached_in_ram: 0,
            spilled_to_disk: 0,
            overflowed: 0,
            ram_hits: 0,
            disk_hits: 0,
            misses: 0,
            spill_errors: 0,
            unpack_errors: 0,
            ram_used: 0,
            disk_used: 0,
        });
        let in_ram = stats.cached_in_ram > before.cached_in_ram;
        let on_disk = stats.spilled_to_disk > before.spilled_to_disk;
        if in_ram {
            self.ram_entry = stats.ram_used.saturating_sub(before.ram_used);
        }
        if on_disk {
            self.disk_entry = stats.disk_used.saturating_sub(before.disk_used);
        }
        self.stored.push(in_ram || on_disk);
        self.last = Some(stats);
    }

    /// Whether GOP `gop` (already observed) is in the cache.
    pub(crate) fn stored(&self, gop: usize) -> bool {
        self.stored.get(gop).copied().unwrap_or(false)
    }

    /// Frames of the observed GOPs the cache did not store.
    pub(crate) fn unstored_frames(&self) -> f64 {
        self.stored
            .iter()
            .zip(&self.gop_frames)
            .filter(|(stored, _)| !**stored)
            .map(|(_, &f)| f as f64)
            .sum()
    }

    /// [`unstored_frames`](Self::unstored_frames) plus the frames of the GOPs
    /// still to come that the cache's remaining room will not hold.
    pub(crate) fn predicted_recompute_frames(&self) -> f64 {
        let Some(last) = self.last else {
            return 0.0;
        };
        let rest = &self.gop_frames[self.stored.len().min(self.gop_frames.len())..];
        let fit = if self.stored.iter().any(|s| !s) {
            0
        } else {
            let ram_fit = match self.ram_entry {
                0 => 0,
                e => self.ram_budget.saturating_sub(last.ram_used) / e,
            };
            // Until a spill is seen, a RAM entry stands in for a disk one
            // (the disk form is somewhat smaller, so this errs pessimistic).
            let disk_entry = if self.disk_entry > 0 { self.disk_entry } else { self.ram_entry };
            let disk_fit = match disk_entry {
                0 => 0,
                e => self.disk_budget.saturating_sub(last.disk_used) / e,
            };
            ram_fit + disk_fit
        };
        self.unstored_frames() + rest.iter().skip(fit).map(|&f| f as f64).sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(cached_in_ram: usize, spilled_to_disk: usize, ram_used: usize, disk_used: usize) -> GopCacheStats {
        GopCacheStats {
            cached_in_ram,
            spilled_to_disk,
            overflowed: 0,
            ram_hits: 0,
            disk_hits: 0,
            misses: 0,
            spill_errors: 0,
            unpack_errors: 0,
            ram_used,
            disk_used,
        }
    }

    #[test]
    fn fraction_tracks_work_when_the_total_is_exact() {
        let mut p = WorkProgress::new(100.0);
        assert_eq!(p.advance(25.0), 0.25);
        assert_eq!(p.advance(25.0), 0.5);
        assert_eq!(p.finish(), 1.0);
    }

    #[test]
    fn raising_the_total_never_moves_the_bar_back() {
        let mut p = WorkProgress::new(100.0);
        assert_eq!(p.advance(50.0), 0.5);
        p.set_total(200.0);
        assert_eq!(p.advance(0.0), 0.5);
        // 150 work left over the remaining half of the bar.
        assert_eq!(p.advance(75.0), 0.75);
    }

    #[test]
    fn lowering_the_total_speeds_up_but_only_finish_reaches_one() {
        let mut p = WorkProgress::new(100.0);
        assert_eq!(p.advance(50.0), 0.5);
        p.set_total(60.0);
        assert_eq!(p.advance(5.0), 0.75);
        let near = p.advance(50.0); // past the estimate
        assert!(near < 1.0 && near >= 0.75, "{near}");
        assert_eq!(p.finish(), 1.0);
    }

    #[test]
    fn ample_cache_predicts_no_recompute() {
        let mut f = RecomputeForecast::new(1_000, 0, vec![30; 4]);
        assert_eq!(f.predicted_recompute_frames(), 0.0);
        f.observe(stats(1, 0, 100, 0));
        assert_eq!(f.predicted_recompute_frames(), 0.0);
    }

    #[test]
    fn ram_then_disk_then_recompute() {
        // GOP 0 took 100 of a 250 RAM budget: one more fits in RAM, three on
        // the 300 disk budget (the RAM size stands in until a spill is seen),
        // so 5 of the 9 GOPs still to come will be re-encoded.
        let mut f = RecomputeForecast::new(250, 300, vec![30; 10]);
        f.observe(stats(1, 0, 100, 0));
        assert_eq!(f.predicted_recompute_frames(), 5.0 * 30.0);
    }

    #[test]
    fn after_an_overflow_every_remaining_gop_is_predicted() {
        let mut f = RecomputeForecast::new(100, 0, vec![30; 10]);
        f.observe(stats(1, 0, 100, 0));
        f.observe(stats(1, 0, 100, 0)); // GOP 1 not stored
        assert!(!f.stored(1));
        assert_eq!(f.unstored_frames(), 30.0);
        assert_eq!(f.predicted_recompute_frames(), 30.0 + 8.0 * 30.0);
    }

    #[test]
    fn disabled_cache_recomputes_every_frame_including_the_tail() {
        let mut f = RecomputeForecast::new(0, 0, vec![30, 30, 3]);
        for _ in 0..3 {
            f.observe(stats(0, 0, 0, 0));
        }
        assert_eq!(f.unstored_frames(), 63.0);
        assert_eq!(f.predicted_recompute_frames(), 63.0);
    }
}
