// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Tier 1 override primitives for the AV1 Pass 2 cached-replay flow.
//!
//! Public surface (consumed by `orchestrator`, `whole_video`, and
//! cost-model tests):
//!
//! - [`OverrideMap`] — sparse position → bit-value plan the STC pass
//!   writes into, consulted at replay time to flip the natural L(1)
//!   bit at planned cover positions.
//! - [`CoverPosition`] / [`CoverPositions`] — typed view over
//!   `WriterRecorder::phasm_bit_positions() + phasm_bit_tags()`,
//!   used for STC plan computation + cursor-parity verification.
//! - [`replay_with_overrides`] / [`finalize_replay`] — hybrid Pass 2
//!   that streams recorded storage tuples through a sink while
//!   applying overrides from an `OverrideMap` at the indexed bit
//!   positions. Architecture per
//!   `docs/design/video/av1/streaming-session.md` § 1.1 Option E.
//!
//! The earlier `WriterStego` live-intercept writer (Writer +
//! StorageBackend dual-impl) was deleted 2026-10-18 — it was always
//! marked as a skeleton (see mod.rs history) and production entirely
//! uses `replay_with_overrides`, which doesn't need live interception.
//! Full record in git history (search `WriterStego`).

use std::collections::HashMap;

use phasm_rav1e::ec::{StorageBackend, WriterBase, WriterEncoder, WriterRecorder};

/// Tier 1 override plan keyed by canonical cover position.
///
/// Sparse position → bit-value map. Real STC integration
/// (cost-weighted position selection over the J-UNIWARD-derived
/// cost vector per cost-model.md § 2) plugs into the
/// [`OverrideMap::from_stc_plan`] constructor in a future commit
/// — for now this is a directly-populated mechanism for round-trip
/// + override-correctness testing.
///
/// Backed by `HashMap<u64, u16>` for O(1) lookup at every
/// `Writer::bit()` call — necessary for 1080p frames with ~200k
/// candidate positions per cost-model.md § 3.
#[derive(Debug, Default, Clone)]
pub struct OverrideMap {
    entries: HashMap<u64, u16>,
}

impl OverrideMap {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set(&mut self, position: u64, override_bit: u16) {
        self.entries.insert(position, override_bit);
    }
    pub fn get(&self, position: u64) -> Option<u16> {
        self.entries.get(&position).copied()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

// === Encoder-side cover positions ===
//
// Typed view over `recorder.phasm_bit_positions()` + `phasm_bit_tags()`.
// Pairs them with a monotonic cursor + the encoder-side tag, so STC
// plan computation + cursor-parity verification can iterate
// structured records instead of raw parallel arrays.
//
// Mirror of `DecodedCoverPosition` in `decoder.rs` on the decode
// side. The round-trip invariant we care about:
//
//   encoder_positions[N].natural_value == decoder_positions[N].decoded_value
//   encoder_positions[N].tag           == decoder_positions[N].tag
//
// for every N — verified by round-trip tests in `decoder.rs`.

/// A single L(1) cover position from a recorded encoder pass. Cursor
/// is the monotonic 50/50 emission index starting at 0; matches the
/// OverrideMap key space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoverPosition {
    /// Monotonic L(1) emission count — matches the OverrideMap key
    /// AND the decoder-side cursor (cursor-parity invariant).
    pub cursor: u64,
    /// Index into `recorder.phasm_storage()` where the matching
    /// `(fl, fh, nms)` tuple lives. Used by `replay_with_overrides`.
    pub storage_index: u32,
    /// The natural bit value at this position (0 or 1) — what the
    /// encoder would emit absent any stego override.
    pub natural_value: u16,
    /// Channel identity:
    /// `phasm_rav1e::phasm_stego::PHASM_TAG_*` constants.
    pub tag: u8,
}

/// All L(1) cover positions captured by an encoder pass. Output of
/// [`CoverPositions::from_recorder`].
#[derive(Debug, Clone, Default)]
pub struct CoverPositions {
    all: Vec<CoverPosition>,
}

impl CoverPositions {
    /// Build a `CoverPositions` view from a post-Pass-1
    /// `WriterRecorder`'s parallel position + tag arrays. Both
    /// slices must have equal length (fork-side invariant).
    pub fn from_recorder(
        bit_positions: &[(u32, u16)],
        bit_tags: &[u8],
    ) -> Self {
        assert_eq!(
            bit_positions.len(),
            bit_tags.len(),
            "phasm-rav1e fork invariant: phasm_bit_positions.len() == phasm_bit_tags.len()"
        );
        let all = bit_positions
            .iter()
            .zip(bit_tags.iter())
            .enumerate()
            .map(|(cursor, (&(idx, val), &tag))| CoverPosition {
                cursor: cursor as u64,
                storage_index: idx,
                natural_value: val,
                tag,
            })
            .collect();
        Self { all }
    }

    /// Iterate all cover positions in emission order.
    pub fn iter(&self) -> impl Iterator<Item = &CoverPosition> {
        self.all.iter()
    }

    pub fn len(&self) -> usize {
        self.all.len()
    }

    pub fn is_empty(&self) -> bool {
        self.all.is_empty()
    }

    /// Count cover positions matching a specific channel tag.
    pub fn count_by_tag(&self, tag: u8) -> usize {
        self.all.iter().filter(|p| p.tag == tag).count()
    }

    /// All positions matching a channel tag, in emission order.
    /// Returns references to the inner positions.
    pub fn filter_by_tag(&self, tag: u8) -> Vec<&CoverPosition> {
        self.all.iter().filter(|p| p.tag == tag).collect()
    }

    /// Borrow the raw position slice.
    pub fn as_slice(&self) -> &[CoverPosition] {
        &self.all
    }
}

// === Hybrid Pass 2 replay-with-overrides ===
//
// Pairs phasm-rav1e's fork patch with phasm-core's STC override
// plan to apply Tier 1 L(1) overrides at REPLAY time (no full re-
// encode required). Architecture per
// `docs/design/video/av1/streaming-session.md` § 1.1 Option E.
//
// Pass 1 (separate orchestration, not in this module): encode through
// `WriterBase<WriterRecorder>`; the recorder captures `storage`
// (every (fl, fh, nms) tuple emitted) and a parallel
// `phasm_bit_positions` index marking which storage entries
// correspond to 50/50 binary emissions (the L(1) cover positions).
//
// Pass 2 (this module): walk `storage` while peeking at
// `phasm_bit_positions`. At each marked index, consult the
// `OverrideMap` keyed by monotonic L(1) cursor; if an override is
// planned, recompute (fl, fh, nms) for the flipped bit and emit that
// instead. Non-bit entries pass through unchanged.

/// Compute the (fl, fh, nms) tuple that rav1e's `Writer::bool(val,
/// 16384)` produces for the given binary value.
///
/// Derived from rav1e `src/ec.rs::485-489` + `src/ec.rs::520-537`:
///
/// ```text
/// bool(val, 16384) → symbol(val as u32, &[16384, 0])
///   with cdf.len() = 2, EC_PROB_SHIFT-encoded:
///     - s=0 (val=false): nms = 2, fl = 32768, fh = cdf[0] = 16384
///     - s=1 (val=true):  nms = 1, fl = cdf[0] = 16384, fh = cdf[1] = 0
/// ```
///
/// Used by [`replay_with_overrides`] to substitute the override tuple
/// at a planned L(1) position without re-running the encoder.
#[inline]
fn bit_to_store_params(bit: u16) -> (u16, u16, u16) {
    match bit {
        0 => (32768, 16384, 2),
        _ => (16384, 0, 1),
    }
}

/// Hybrid Pass 2 replay: stream recorded storage tuples to `dest` and
/// apply Tier 1 L(1) overrides at the indexed bit positions.
///
/// # Arguments
/// - `storage` — `(fl, fh, nms)` tuples recorded by Pass 1.
///   Obtain via [`WriterBase<WriterRecorder>::phasm_storage`].
/// - `bit_positions` — sorted-ascending `(storage_index,
///   natural_bit_value)` index of 50/50 emissions. Obtain via
///   [`WriterBase<WriterRecorder>::phasm_bit_positions`].
/// - `plan` — Tier 1 override map keyed by monotonic L(1) cursor (0,
///   1, 2, … in emission order, NOT by storage_index).
/// - `dest` — sink to emit the (possibly overridden) tuple stream to.
///   Typically a `WriterBase<WriterEncoder>`.
///
/// # Semantics
/// Iterates `storage` in order. At each entry, checks whether the
/// current `storage` index appears next in `bit_positions`. If so,
/// increments the L(1) cursor and looks up the override; if planned,
/// substitutes [`bit_to_store_params`] for the flipped value. Other
/// entries pass through to `dest.store(fl, fh, nms)` unchanged.
///
/// # Determinism
/// With an empty plan, output bytes are byte-identical to a direct
/// Pass-1 encode of the same input via `WriterBase<WriterEncoder>`.
/// Verified by [`tests::hybrid_replay_empty_plan_byte_identity`].
pub fn replay_with_overrides<S: StorageBackend>(
    storage: &[(u16, u16, u16)],
    bit_positions: &[(u32, u16)],
    plan: &OverrideMap,
    dest: &mut S,
) {
    let mut bit_iter = bit_positions.iter().peekable();
    let mut cursor: u64 = 0;
    for (i, &(fl, fh, nms)) in storage.iter().enumerate() {
        if let Some(&&(bit_idx, natural)) = bit_iter.peek() {
            if bit_idx as usize == i {
                bit_iter.next();
                let effective = plan.get(cursor).unwrap_or(natural);
                cursor += 1;
                let (fl_e, fh_e, nms_e) = bit_to_store_params(effective);
                dest.store(fl_e, fh_e, nms_e);
                continue;
            }
        }
        dest.store(fl, fh, nms);
    }
}

/// Convenience wrapper: drive [`replay_with_overrides`] from a
/// post-Pass-1 [`WriterBase<WriterRecorder>`] and a [`WriterEncoder`]
/// sink, returning the finalized bitstream bytes.
pub fn finalize_replay(
    recorder: &WriterBase<WriterRecorder>,
    plan: &OverrideMap,
) -> Vec<u8> {
    let mut sink = WriterEncoder::new();
    replay_with_overrides(
        recorder.phasm_storage(),
        recorder.phasm_bit_positions(),
        plan,
        &mut sink,
    );
    sink.done()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_map_set_get() {
        let mut plan = OverrideMap::new();
        plan.set(5, 1);
        plan.set(10, 0);
        assert_eq!(plan.get(5), Some(1));
        assert_eq!(plan.get(10), Some(0));
        assert_eq!(plan.get(7), None);
    }

    #[test]
    fn override_map_hashmap_o1_lookup() {
        let mut plan = OverrideMap::new();
        for i in 0..1000u64 {
            plan.set(i, (i & 1) as u16);
        }
        assert_eq!(plan.len(), 1000);
        assert_eq!(plan.get(0), Some(0));
        assert_eq!(plan.get(1), Some(1));
        assert_eq!(plan.get(999), Some(1));
        assert_eq!(plan.get(1000), None);
    }

    #[test]
    fn override_map_overwrites_on_duplicate_set() {
        let mut plan = OverrideMap::new();
        plan.set(42, 0);
        plan.set(42, 1);
        assert_eq!(plan.get(42), Some(1));
        assert_eq!(plan.len(), 1);
    }

    #[test]
    fn bit_to_store_params_table() {
        // Mirrors rav1e's `symbol(val, &[16384, 0])` output for bool(val, 16384).
        assert_eq!(bit_to_store_params(0), (32768, 16384, 2));
        assert_eq!(bit_to_store_params(1), (16384, 0, 1));
    }

    /// Empty plan: `replay_with_overrides` passes recorded storage
    /// tuples through unchanged, byte-identical to a direct replay
    /// into a `WriterEncoder` sink.
    #[test]
    fn replay_with_empty_plan_is_byte_identical() {
        use phasm_rav1e::ec::{Writer, WriterRecorder as RavWriterRecorder};

        // Build a recording.
        let mut rec = RavWriterRecorder::new();
        for i in 0..16 {
            rec.bit((i & 1) as u16);
        }
        rec.literal(8, 0xab);

        // Native replay through WriterEncoder.
        let mut sink_a = WriterEncoder::new();
        rec.replay(&mut sink_a);
        let bytes_a = sink_a.done();

        // Replay through replay_with_overrides with empty plan.
        let mut rec2 = RavWriterRecorder::new();
        for i in 0..16 {
            rec2.bit((i & 1) as u16);
        }
        rec2.literal(8, 0xab);
        let plan = OverrideMap::new();
        let bytes_b = finalize_replay(&rec2, &plan);

        assert_eq!(
            bytes_a, bytes_b,
            "replay_with_overrides with empty plan must be byte-identical to a native replay"
        );
    }

    /// Non-empty plan at a planned L(1) cover position produces
    /// different bytes than empty-plan replay — proves the override
    /// actually took effect on the wire.
    #[test]
    fn replay_with_override_changes_bytes() {
        use phasm_rav1e::ec::{Writer, WriterRecorder as RavWriterRecorder};

        let build = || {
            let mut r = RavWriterRecorder::new();
            for i in 0..16u32 {
                r.bit((i & 1) as u16);
            }
            r
        };

        let rec_a = build();
        let bytes_natural = finalize_replay(&rec_a, &OverrideMap::new());

        let rec_b = build();
        let mut plan = OverrideMap::new();
        plan.set(3, 1 - ((3u16) & 1)); // flip the bit at cursor 3
        let bytes_overridden = finalize_replay(&rec_b, &plan);

        assert_ne!(
            bytes_natural, bytes_overridden,
            "override at a planned cursor must change the bitstream"
        );
    }

    #[test]
    fn cover_positions_from_recorder_pairs_tags_and_positions() {
        // phasm-rav1e fork invariant check: bit_positions.len() ==
        // bit_tags.len(); the typed view pairs them 1:1 in emission
        // order.
        let bit_positions = vec![(10u32, 0u16), (12, 1), (15, 0)];
        let bit_tags = vec![1u8, 2, 1];
        let cps = CoverPositions::from_recorder(&bit_positions, &bit_tags);
        assert_eq!(cps.len(), 3);
        assert!(!cps.is_empty());
        let v: Vec<_> = cps.iter().copied().collect();
        assert_eq!(v[0].cursor, 0);
        assert_eq!(v[0].storage_index, 10);
        assert_eq!(v[0].natural_value, 0);
        assert_eq!(v[0].tag, 1);
        assert_eq!(cps.count_by_tag(1), 2);
        assert_eq!(cps.count_by_tag(2), 1);
    }
}
