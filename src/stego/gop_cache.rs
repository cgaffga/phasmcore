// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Bounded per-GOP cache with RAM tier, disk spill, and overflow-to-recompute.
//!
//! Both video stego pipelines visit each GOP more than once and recompute the
//! same expensive prefix each time — AV1's cascade re-runs Pass 1 per parity
//! tier; H.264's shadow encode runs an identical clean-encode → walk →
//! content-costs → gate sequence in both sweeps. This caches that work.
//!
//! ## The load-bearing property: a miss costs only recompute
//!
//! This cache is an **optimization, never a correctness requirement**. Every
//! path that fails — budget exhausted, scratch dir unavailable, disk full,
//! corrupt spill file — degrades to "not cached", and the caller recomputes.
//! That is what makes it safe to add to a pipeline whose whole point is
//! bounded memory:
//!
//! * it cannot OOM (the RAM tier is a hard byte cap),
//! * it cannot fill the disk (the disk tier is a hard byte cap),
//! * it cannot change output (a hit and a recompute are the same bytes),
//! * it cannot fail an encode (see [`BoundedGopCache::insert`]).
//!
//! **This is the distinction from the `DecisionCacheSpill` / `YuvSpill`
//! machinery that WV.6.g deliberately deleted** (`8f8d938e`). Those were
//! unbounded and load-bearing — the encode did not work without them. Anyone
//! reading this as a reversal of that rearchitecture should read this paragraph
//! first.
//!
//! Two consequences worth stating, because they are easy to "harden" away:
//!
//! * On mobile the spill belongs in **purgeable** storage (iOS `Caches`,
//!   Android `cacheDir`). An OS reclaim mid-encode costs latency, not
//!   correctness. This is the one video artifact that *should* live there —
//!   unlike the encode source, which had to be moved out of a purgeable
//!   directory precisely because losing it broke the encode.
//! * Budget exhaustion is not an error. It is the design working.
//!
//! ## Budgets come from the caller
//!
//! This module reads **no environment variables** and has no notion of a
//! default. Budgets are parameters, because the right numbers differ by an
//! order of magnitude between a desktop CLI run and a phone — see
//! `docs/design/video/gop-cache-budgets.md`. Keeping the policy at the call
//! site is what avoids a `#[cfg(target_os)]` thicket in here.
//!
//! Budgets are in **bytes, not GOP counts**: a 480p GOP is roughly a ninth of
//! a 1080p one, so byte budgets adapt to resolution on their own.

use std::io;
use std::path::{Path, PathBuf};

/// A per-GOP payload that [`BoundedGopCache`] can hold in RAM or spill.
///
/// Implementors own their serialization. A hand-rolled flat format is the
/// convention here (the crate carries no serde/bincode dependency); it only
/// has to round-trip within one process run, so there is no compatibility
/// obligation — but version the format anyway so a stale file hard-fails
/// instead of unpacking into garbage.
///
/// `Clone` is required because a RAM hit hands the caller an owned copy while
/// the cache keeps its own (the same GOP is fetched again — AV1 per parity
/// tier, H.264 per sweep). Serializing on a RAM hit instead would defeat the
/// point of having a RAM tier at all.
pub trait GopCachePayload: Sized + Clone {
    /// Approximate heap footprint, used for budget accounting. Over-estimating
    /// is safe (the cache simply holds less); under-estimating lets the RAM
    /// tier exceed its budget, which is the one thing this type must not do.
    fn cached_size(&self) -> usize;

    /// Serialize for the disk tier.
    fn flat_pack(&self) -> Vec<u8>;

    /// Inverse of [`flat_pack`](GopCachePayload::flat_pack). An `Err` is
    /// treated as a cache miss, so returning one is always safe.
    fn flat_unpack(bytes: &[u8]) -> Result<Self, String>;
}

enum Entry<T> {
    Absent,
    InRam(T),
    OnDisk(PathBuf),
}

/// Per-GOP cache: insert as each GOP is first computed, fetch on every later
/// visit. Spills to disk when the RAM budget is exhausted, and gives up
/// (leaving the caller to recompute) when both budgets are.
///
/// The scratch directory is created lazily — a run that never spills never
/// touches the filesystem — and removed on `Drop`.
pub struct BoundedGopCache<T: GopCachePayload> {
    entries: Vec<Entry<T>>,
    ram_budget: usize,
    disk_budget: usize,
    ram_used: usize,
    disk_used: usize,
    scratch_dir: Option<PathBuf>,
    scratch_dir_parent: PathBuf,
    /// Distinguishes concurrent users' scratch directories.
    tag: &'static str,
    cached_in_ram: usize,
    spilled_to_disk: usize,
    overflowed: usize,
    ram_hits: usize,
    disk_hits: usize,
    misses: usize,
    /// Spill writes that failed (disk full, permissions, …). Counted rather
    /// than propagated — see [`insert`](Self::insert).
    spill_errors: usize,
    /// Spill reads that failed or unpacked wrong. Also counted, not propagated.
    unpack_errors: usize,
}

/// What a cache run did. Purely diagnostic — no caller should branch on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GopCacheStats {
    pub cached_in_ram: usize,
    pub spilled_to_disk: usize,
    pub overflowed: usize,
    pub ram_hits: usize,
    pub disk_hits: usize,
    pub misses: usize,
    pub spill_errors: usize,
    pub unpack_errors: usize,
    pub ram_used: usize,
    pub disk_used: usize,
}

impl<T: GopCachePayload> BoundedGopCache<T> {
    /// `ram_budget` / `disk_budget` are byte caps; either may be 0 to disable
    /// that tier. `0 / 0` makes every fetch a miss, which is a valid way to
    /// turn the cache off without branching at the call sites.
    pub fn new(
        n_gops: usize,
        ram_budget: usize,
        disk_budget: usize,
        scratch_dir_parent: PathBuf,
        tag: &'static str,
    ) -> Self {
        let mut entries = Vec::with_capacity(n_gops);
        for _ in 0..n_gops {
            entries.push(Entry::Absent);
        }
        Self {
            entries,
            ram_budget,
            disk_budget,
            ram_used: 0,
            disk_used: 0,
            scratch_dir: None,
            scratch_dir_parent,
            tag,
            cached_in_ram: 0,
            spilled_to_disk: 0,
            overflowed: 0,
            ram_hits: 0,
            disk_hits: 0,
            misses: 0,
            spill_errors: 0,
            unpack_errors: 0,
        }
    }

    fn ensure_scratch_dir(&mut self) -> io::Result<&Path> {
        if self.scratch_dir.is_none() {
            // pid + nanos: two concurrent runs (or two caches in one run with
            // different tags) must not share a directory, because `Drop`
            // removes the whole tree.
            let pid = std::process::id();
            let ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let dir = self
                .scratch_dir_parent
                .join(format!("{}{pid}-{ns}", scratch_prefix(self.tag)));
            std::fs::create_dir_all(&dir)?;
            self.scratch_dir = Some(dir);
        }
        Ok(self.scratch_dir.as_deref().unwrap())
    }

    /// Offer `payload` for GOP `gop_idx`. Policy:
    ///
    /// 1. fits the remaining RAM budget → hold in RAM;
    /// 2. else fits the remaining disk budget → flat-pack to a scratch file;
    /// 3. else → drop it; the caller will recompute on fetch.
    ///
    /// **Infallible by design.** Failing to *cache* must never fail an encode,
    /// so an unwritable scratch directory or a short write is counted
    /// (`spill_errors`) and the entry is simply left absent. The AV1 original
    /// propagated these with `?`, which meant a transient disk problem aborted
    /// a long encode that would otherwise have completed slightly slower —
    /// backwards for something whose entire purpose is to save time.
    ///
    /// Out-of-range `gop_idx` is ignored rather than panicking: a caller that
    /// miscounts GOPs gets a slow encode, not a crash.
    pub fn insert(&mut self, gop_idx: usize, payload: T) {
        if gop_idx >= self.entries.len() {
            return;
        }
        let size = payload.cached_size();

        if self.ram_budget > 0 && self.ram_used.saturating_add(size) <= self.ram_budget {
            self.ram_used = self.ram_used.saturating_add(size);
            self.cached_in_ram += 1;
            self.entries[gop_idx] = Entry::InRam(payload);
            return;
        }

        if self.disk_budget > 0 && self.disk_used.saturating_add(size) <= self.disk_budget {
            let bytes = payload.flat_pack();
            match self
                .ensure_scratch_dir()
                .map(|d| d.join(format!("gop_{gop_idx:08}.bin")))
                .and_then(|path| std::fs::write(&path, &bytes).map(|()| path))
            {
                Ok(path) => {
                    self.disk_used = self.disk_used.saturating_add(bytes.len());
                    self.spilled_to_disk += 1;
                    self.entries[gop_idx] = Entry::OnDisk(path);
                }
                Err(_) => {
                    self.spill_errors += 1;
                }
            }
            return;
        }

        self.overflowed += 1;
    }

    /// Fetch GOP `gop_idx`, or `None` if the caller must recompute.
    ///
    /// Every failure mode returns `None`: never cached, over budget, missing
    /// or unreadable spill file, corrupt payload. There is deliberately no
    /// error channel — a cache miss is the only thing a caller can do about
    /// any of them.
    ///
    /// A RAM hit clones, because the same GOP can be fetched again (AV1 per
    /// parity tier, H.264 per sweep) and the cache keeps ownership. The clone
    /// is dominated by the encode it avoids.
    pub fn fetch(&mut self, gop_idx: usize) -> Option<T> {
        if gop_idx >= self.entries.len() {
            self.misses += 1;
            return None;
        }
        match &self.entries[gop_idx] {
            Entry::Absent => {
                self.misses += 1;
                None
            }
            Entry::InRam(p) => {
                // Clone, keeping the cached copy: the same GOP is fetched
                // again (AV1 per parity tier, H.264 per sweep). The clone is
                // dominated by the encode it avoids.
                let v = p.clone();
                self.ram_hits += 1;
                Some(v)
            }
            Entry::OnDisk(path) => {
                let path = path.clone();
                match std::fs::read(&path).map_err(|e| e.to_string()).and_then(|b| T::flat_unpack(&b))
                {
                    Ok(v) => {
                        self.disk_hits += 1;
                        Some(v)
                    }
                    Err(_) => {
                        self.unpack_errors += 1;
                        self.misses += 1;
                        None
                    }
                }
            }
        }
    }

    pub fn stats(&self) -> GopCacheStats {
        GopCacheStats {
            cached_in_ram: self.cached_in_ram,
            spilled_to_disk: self.spilled_to_disk,
            overflowed: self.overflowed,
            ram_hits: self.ram_hits,
            disk_hits: self.disk_hits,
            misses: self.misses,
            spill_errors: self.spill_errors,
            unpack_errors: self.unpack_errors,
            ram_used: self.ram_used,
            disk_used: self.disk_used,
        }
    }

    /// True when the cache spilled at least once — useful for a one-line
    /// summary in a perf trace.
    pub fn spilled(&self) -> bool {
        self.spilled_to_disk > 0
    }
}

impl<T: GopCachePayload> Drop for BoundedGopCache<T> {
    /// Normal-path cleanup. **Not sufficient on its own** — see
    /// [`sweep_orphans`], which is the primary mechanism on mobile.
    fn drop(&mut self) {
        if let Some(dir) = &self.scratch_dir {
            let _ = std::fs::remove_dir_all(dir); // best-effort
        }
    }
}

/// Default `max_age` for [`sweep_orphans`]. Six hours is far longer than any
/// plausible single encode (a 10-minute 1080p shadow clip is minutes, not
/// hours) while still reclaiming debris the same day.
pub const ORPHAN_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(6 * 60 * 60);

/// Directory-name prefix for a cache's scratch tree. Shared by
/// [`BoundedGopCache::ensure_scratch_dir`] and [`sweep_orphans`]; changing it
/// in one place without the other orphans every existing directory forever.
fn scratch_prefix(tag: &str) -> String {
    format!("phasm-{tag}-gopcache-")
}

/// Delete scratch trees left behind by a previous run.
///
/// **Why this is required and not merely tidy.** `Drop` covers normal
/// completion and unwinding error paths, but three real cases skip it
/// entirely:
///
/// * **`panic = "abort"`.** The `mobile-ship` profile both mobile bridges ship
///   with sets it, so on a phone *any* panic terminates without unwinding and
///   `Drop` never runs.
/// * **SIGKILL.** iOS jetsam under memory pressure, Android's Task Manager
///   "Stop" (which Google documents as giving no callback), an OEM killer, or
///   the user force-quitting mid-encode.
/// * **Power loss / reboot** during a long encode.
///
/// Each of those strands up to the full disk budget. At the mobile cap that is
/// up to 2 GB of a user's storage, and at the desktop cap up to 8 GB, with
/// nothing that would ever reclaim it. Hence: call this at the start of every
/// encode, **and** from the host app's launch path.
///
/// ## Not deleting a live cache
///
/// Two guards, and deliberately **no pid-liveness check**:
///
/// * **own pid** — this process's own directory is always skipped, so calling
///   the sweep at the start of an encode cannot touch the cache that encode is
///   about to populate.
/// * **age** — anything older than `max_age` is debris. Give it a value
///   comfortably longer than the longest plausible encode.
///
/// A `kill(pid, 0)` liveness probe would be the textbook third guard, and the
/// first draft of this had one. It was removed because `libc` is a
/// dev-dependency here and `phasm-core` is published — adding a runtime
/// dependency to every downstream consumer to slightly sharpen a *cleanup*
/// heuristic is a bad trade.
///
/// It is also unnecessary, and the reason is the invariant at the top of this
/// module: **the worst case of deleting a live cache is a cache miss.** A
/// concurrent encode in another process whose directory somehow aged past
/// `max_age` would recompute the affected GOPs and produce byte-identical
/// output, slightly slower. There is no correctness exposure to protect
/// against, so the cheapest adequate guard is the right one.
///
/// Returns how many directories were removed. Never fails: an unreadable
/// parent, a permission error or a vanished directory are all skipped, because
/// a sweep that could fail an encode would defeat the purpose.
pub fn sweep_orphans(parent: &Path, tag: &str, max_age: std::time::Duration) -> usize {
    let prefix = scratch_prefix(tag);
    let self_pid = std::process::id();
    let Ok(entries) = std::fs::read_dir(parent) else {
        return 0;
    };
    let now = std::time::SystemTime::now();
    let mut removed = 0usize;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(rest) = name.strip_prefix(&prefix) else { continue };

        // Never touch our own tree; the live cache owns it.
        let owner_pid: Option<u32> = rest.split('-').next().and_then(|p| p.parse().ok());
        if owner_pid == Some(self_pid) {
            continue;
        }

        // Unreadable metadata is treated as age 0 — i.e. left alone. Better to
        // leak a directory this pass than to delete one we know nothing about.
        let age = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| now.duration_since(t).ok())
            .unwrap_or(std::time::Duration::ZERO);

        if age > max_age && std::fs::remove_dir_all(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, PartialEq, Debug)]
    struct Blob(Vec<u8>);

    impl GopCachePayload for Blob {
        fn cached_size(&self) -> usize {
            self.0.len()
        }
        fn flat_pack(&self) -> Vec<u8> {
            self.0.clone()
        }
        fn flat_unpack(bytes: &[u8]) -> Result<Self, String> {
            Ok(Blob(bytes.to_vec()))
        }
    }

    fn scratch() -> PathBuf {
        std::env::var("PHASM_TMP")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp"))
    }

    #[test]
    fn ram_tier_holds_within_budget_and_round_trips() {
        let mut c: BoundedGopCache<Blob> =
            BoundedGopCache::new(4, 1024, 0, scratch(), "test-ram");
        c.insert(0, Blob(vec![7; 100]));
        assert_eq!(c.fetch(0), Some(Blob(vec![7; 100])));
        let s = c.stats();
        assert_eq!((s.cached_in_ram, s.spilled_to_disk, s.overflowed), (1, 0, 0));
    }

    #[test]
    fn spills_to_disk_when_ram_is_exhausted() {
        let mut c: BoundedGopCache<Blob> =
            BoundedGopCache::new(4, 10, 1024, scratch(), "test-spill");
        c.insert(0, Blob(vec![1; 100])); // too big for RAM → disk
        assert!(c.spilled());
        assert_eq!(c.fetch(0), Some(Blob(vec![1; 100])));
        assert_eq!(c.stats().disk_hits, 1);
    }

    /// The property the whole design rests on: when both budgets are gone the
    /// cache reports a miss rather than erroring, and the caller recomputes.
    #[test]
    fn overflow_is_a_miss_not_an_error() {
        let mut c: BoundedGopCache<Blob> =
            BoundedGopCache::new(2, 0, 0, scratch(), "test-overflow");
        c.insert(0, Blob(vec![9; 64]));
        assert_eq!(c.fetch(0), None, "no budget ⇒ miss");
        let s = c.stats();
        assert_eq!(s.overflowed, 1);
        assert_eq!(s.spill_errors, 0, "overflow is not a failure");
    }

    /// A corrupt spill file must read as a miss, not propagate.
    #[test]
    fn corrupt_spill_file_reads_as_a_miss() {
        #[derive(Clone)]
        struct Strict;
        impl GopCachePayload for Strict {
            fn cached_size(&self) -> usize {
                8
            }
            fn flat_pack(&self) -> Vec<u8> {
                vec![0xAB; 8]
            }
            fn flat_unpack(bytes: &[u8]) -> Result<Self, String> {
                if bytes == [0xAB; 8] {
                    Ok(Strict)
                } else {
                    Err("bad magic".into())
                }
            }
        }
        let mut c: BoundedGopCache<Strict> =
            BoundedGopCache::new(2, 0, 4096, scratch(), "test-corrupt");
        c.insert(0, Strict);
        assert!(c.spilled());
        // Corrupt the spill file behind the cache's back.
        if let Entry::OnDisk(p) = &c.entries[0] {
            std::fs::write(p, b"garbage!").unwrap();
        } else {
            panic!("expected a disk entry");
        }
        assert!(c.fetch(0).is_none());
        assert_eq!(c.stats().unpack_errors, 1);
    }

    /// Out-of-range indices must not panic — a caller that miscounts GOPs gets
    /// a slow encode, not a crash.
    #[test]
    fn out_of_range_index_is_ignored() {
        let mut c: BoundedGopCache<Blob> =
            BoundedGopCache::new(1, 1024, 1024, scratch(), "test-oob");
        c.insert(99, Blob(vec![1; 8]));
        assert!(c.fetch(99).is_none());
    }

    /// Simulates the kill case: a scratch tree whose owning process is gone.
    /// This is the path that `Drop` cannot cover — `panic = "abort"` on the
    /// mobile-ship profile, jetsam, Task Manager "Stop", reboot.
    #[test]
    fn sweep_reclaims_debris_from_a_dead_run() {
        let parent = scratch().join(format!("sweeptest-{}", std::process::id()));
        std::fs::create_dir_all(&parent).unwrap();

        // A tree that looks like it came from another, long-finished process.
        let orphan = parent.join(format!("{}999999-123", scratch_prefix("h264-shadow")));
        std::fs::create_dir_all(&orphan).unwrap();
        std::fs::write(orphan.join("gop_00000000.bin"), b"debris").unwrap();

        // Zero max_age makes everything older than "now" stale, which is what
        // a real launch sweep sees for debris from a previous run.
        let removed = sweep_orphans(&parent, "h264-shadow", std::time::Duration::ZERO);
        assert_eq!(removed, 1, "the orphan should have been reclaimed");
        assert!(!orphan.exists());

        let _ = std::fs::remove_dir_all(&parent);
    }

    /// The sweep must never delete the cache of the process calling it —
    /// otherwise calling it at the start of an encode would wipe the cache
    /// that encode is about to fill.
    #[test]
    fn sweep_never_touches_its_own_live_cache() {
        let parent = scratch().join(format!("sweepown-{}", std::process::id()));
        std::fs::create_dir_all(&parent).unwrap();

        let mut c: BoundedGopCache<Blob> =
            BoundedGopCache::new(2, 0, 4096, parent.clone(), "h264-shadow");
        c.insert(0, Blob(vec![3; 64]));
        let live = c.scratch_dir.clone().expect("spill should have created one");
        assert!(live.exists());

        // Even with max_age = 0, our own directory is exempt.
        let removed = sweep_orphans(&parent, "h264-shadow", std::time::Duration::ZERO);
        assert_eq!(removed, 0, "must not reclaim the live cache");
        assert!(live.exists());
        assert_eq!(c.fetch(0), Some(Blob(vec![3; 64])), "cache still usable");

        drop(c);
        let _ = std::fs::remove_dir_all(&parent);
    }

    /// A different cache's directories must be left alone — the tag scopes it.
    #[test]
    fn sweep_is_scoped_by_tag() {
        let parent = scratch().join(format!("sweeptag-{}", std::process::id()));
        std::fs::create_dir_all(&parent).unwrap();
        let other = parent.join(format!("{}999999-7", scratch_prefix("av1-pass1")));
        std::fs::create_dir_all(&other).unwrap();

        let removed = sweep_orphans(&parent, "h264-shadow", std::time::Duration::ZERO);
        assert_eq!(removed, 0);
        assert!(other.exists(), "another tag's tree is not ours to delete");

        let _ = std::fs::remove_dir_all(&parent);
    }

    /// A young foreign directory survives — that is what protects a
    /// concurrently-running encode in another process.
    #[test]
    fn sweep_spares_recent_directories() {
        let parent = scratch().join(format!("sweepyoung-{}", std::process::id()));
        std::fs::create_dir_all(&parent).unwrap();
        let young = parent.join(format!("{}999999-9", scratch_prefix("h264-shadow")));
        std::fs::create_dir_all(&young).unwrap();

        let removed = sweep_orphans(&parent, "h264-shadow", ORPHAN_MAX_AGE);
        assert_eq!(removed, 0, "a directory younger than max_age stays");
        assert!(young.exists());

        let _ = std::fs::remove_dir_all(&parent);
    }

    /// A missing or unreadable parent must be a no-op, never a panic — the
    /// sweep runs on a path that may not exist yet.
    #[test]
    fn sweep_on_a_missing_parent_is_a_noop() {
        let missing = scratch().join("definitely-not-here-xyzzy");
        assert_eq!(sweep_orphans(&missing, "h264-shadow", ORPHAN_MAX_AGE), 0);
    }

    #[test]
    fn scratch_dir_is_removed_on_drop() {
        let dir;
        {
            let mut c: BoundedGopCache<Blob> =
                BoundedGopCache::new(2, 0, 4096, scratch(), "test-drop");
            c.insert(0, Blob(vec![5; 32]));
            dir = c.scratch_dir.clone().expect("spill should have created one");
            assert!(dir.exists());
        }
        assert!(!dir.exists(), "Drop must clean the scratch tree");
    }
}
