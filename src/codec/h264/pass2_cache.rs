//! Per-MB decision cache for the Pass-2 replay architecture (#533).
//!
//! Pass-1 captures the encoder's complete per-MB mode decisions
//! (mb_type, sub_mb_type, MVs, ref_idx, intra pred modes, QP delta);
//! Pass-2 replays them via the fork's REPLAY pass mode, bypassing
//! RDO/ME. This locks Pass-2 onto Pass-1's mode-decision domain,
//! collapsing the 4-domain drift bug (#530) into a wire-level
//! bypass-bin override.
//!
//! Scope = one GOP. The streaming session drops the cache between
//! GOPs (memory bound: ~3 MB at 1080p IPPPP, 30-frame GOP).

use std::collections::HashMap;

use crate::codec::h264::openh264::MbDecision;

/// Key identifying one macroblock within a GOP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MbKey {
    pub frame_num: u32,
    pub mb_x: u16,
    pub mb_y: u16,
}

impl MbKey {
    pub fn new(frame_num: u32, mb_x: u16, mb_y: u16) -> Self {
        Self { frame_num, mb_x, mb_y }
    }
}

/// Pass-1 → Pass-2 decision cache. Insertion is O(1); lookup is O(1).
/// One instance per GOP — caller resets at GOP boundary.
#[derive(Debug, Default)]
pub struct DecisionCache {
    entries: HashMap<MbKey, MbDecision>,
}

impl DecisionCache {
    /// New empty cache. Capacity is reserved lazily on first insert.
    pub fn new() -> Self {
        Self::default()
    }

    /// New empty cache with capacity for `mb_count` MBs (one GOP).
    pub fn with_capacity(mb_count: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(mb_count),
        }
    }

    /// Insert a captured decision. Overwrites any existing entry for
    /// the same MB — last writer wins (Pass-1 streams strictly in
    /// encode order so this happens only on re-encode of the same MB).
    pub fn insert(&mut self, decision: MbDecision) {
        let key = MbKey::new(decision.frame_num, decision.mb_x, decision.mb_y);
        self.entries.insert(key, decision);
    }

    /// Look up a decision for replay. Returns `None` on miss — the
    /// replay trampoline interprets that as "fall back to RDO/ME".
    pub fn get(&self, frame_num: u32, mb_x: u16, mb_y: u16) -> Option<MbDecision> {
        self.entries.get(&MbKey::new(frame_num, mb_x, mb_y)).copied()
    }

    /// Number of cached decisions.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the cache holds no decisions.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop all cached decisions. Caller invokes between GOPs to
    /// keep the working set bounded.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_decision(frame_num: u32, mb_x: u16, mb_y: u16) -> MbDecision {
        // SAFETY: MbDecision is repr(C) plain-data; zero-init matches
        // the C "uninitialized → no decision" semantics for fields we
        // don't touch here.
        let mut d: MbDecision = unsafe { core::mem::zeroed() };
        d.frame_num = frame_num;
        d.mb_x = mb_x;
        d.mb_y = mb_y;
        d
    }

    #[test]
    fn insert_then_get() {
        let mut cache = DecisionCache::new();
        cache.insert(dummy_decision(0, 5, 10));
        let got = cache.get(0, 5, 10).expect("cache hit");
        assert_eq!(got.frame_num, 0);
        assert_eq!(got.mb_x, 5);
        assert_eq!(got.mb_y, 10);
    }

    #[test]
    fn miss_returns_none() {
        let cache = DecisionCache::new();
        assert!(cache.get(0, 0, 0).is_none());
    }

    #[test]
    fn clear_drops_entries() {
        let mut cache = DecisionCache::new();
        cache.insert(dummy_decision(0, 0, 0));
        cache.insert(dummy_decision(0, 1, 0));
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }
}
