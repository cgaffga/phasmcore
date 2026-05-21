// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! [`WriterStego`] — Writer + StorageBackend dual-impl that wraps
//! a [`WriterBase<WriterEncoder>`] and intercepts `bit()` calls to
//! apply Tier 1 L(1) overrides per an STC plan.
//!
//! See `docs/design/video/av1/streaming-session.md` § 3.2.
//!
//! Audit B-S4 finding (verified by Q11 spike at
//! `research/16-spike-Q11-rav1e-api.md`): `WriterRecorder::replay`
//! takes `&mut dyn StorageBackend`, not `&mut dyn Writer`. So the
//! stego writer must impl BOTH traits — Writer for live encode
//! interception at `bit()` (Tier 1 L(1) emission), StorageBackend
//! for the replay sink role.
//!
//! # W3.3 STATE (2026-05-21)
//!
//! Real Writer + StorageBackend impls landed; both forward to the
//! inner `WriterBase<WriterEncoder>` via trait disambiguation.
//! `bit()` has the override interception point with a placeholder
//! `OverrideMap` Vec<(u64, u16)> mechanism. STC plan integration
//! (real cost-weighted position selection) is W3.4+.
//!
//! Test coverage includes:
//! - skeleton constructs / defaults
//! - rav1e types importable (compile-time)
//! - no-override path is byte-identical to clean WriterBase<WriterEncoder>
//! - override at a specific position substitutes the bit value

use std::collections::HashMap;

use phasm_rav1e::context::{CDFContext, CDFContextLog, CDFOffset};
use phasm_rav1e::ec::{
    StorageBackend, Writer, WriterBase, WriterCheckpoint, WriterEncoder,
    WriterRecorder,
};

/// Tier 1 override plan keyed by canonical cover position.
///
/// v0.3-AV1 uses a sparse position → bit-value map. Real STC
/// integration (cost-weighted position selection over the
/// J-UNIWARD-derived cost vector per cost-model.md § 2) plugs into
/// the [`OverrideMap::from_stc_plan`] constructor in a future commit
/// — for now this is a directly-populated mechanism for round-trip
/// + override-correctness testing.
///
/// W3.4: switched from `Vec<(u64, u16)>` to `HashMap<u64, u16>` for
/// O(1) lookup at every `Writer::bit()` call (was O(N) in the Vec
/// version, which becomes prohibitive for 1080p frames with ~200k
/// candidate positions per cost-model.md § 3).
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

/// AV1 stego writer — intercepts `Writer::bit()` to apply Tier 1
/// L(1) overrides per the active [`OverrideMap`]; also acts as a
/// [`StorageBackend`] sink for [`phasm_rav1e::ec::WriterRecorder`]
/// replay in Pass 2 cached-replay orchestration.
pub struct WriterStego {
    /// Inner entropy coder — owns range-coder state + produces
    /// final bitstream bytes.
    inner: WriterBase<WriterEncoder>,
    /// Tier 1 override plan (sparse position → bit-value map).
    /// Empty by default → no overrides → bit() forwards unchanged.
    plan: OverrideMap,
    /// Monotonic counter incremented on every `bit()` call.
    /// Position-keyed override lookup uses this. Real cover-position
    /// canonical order (per streaming-session.md § 5) will replace
    /// this with a more structured cursor in W3.4+.
    cursor: u64,
}

impl WriterStego {
    /// Construct a WriterStego with empty override plan and a fresh
    /// [`WriterEncoder`] backing store. Cursor starts at 0.
    pub fn new() -> Self {
        Self {
            inner: WriterEncoder::new(),
            plan: OverrideMap::new(),
            cursor: 0,
        }
    }

    /// Set the active override plan (replacing any prior plan).
    pub fn set_plan(&mut self, plan: OverrideMap) {
        self.plan = plan;
    }

    /// Current cursor position (monotonic count of `bit()` calls).
    pub fn cursor(&self) -> u64 {
        self.cursor
    }

    /// Consume the writer, flush the inner range coder, and return
    /// the final encoded bytes.
    ///
    /// W3.4: now invokes `WriterBase<WriterEncoder>::done()` (rav1e
    /// `src/ec.rs::434`) which finalizes the range coder by emitting
    /// the minimum bits needed for unambiguous decode of all symbols
    /// written so far.
    pub fn finalize(mut self) -> Vec<u8> {
        self.inner.done()
    }
}

impl Default for WriterStego {
    fn default() -> Self {
        Self::new()
    }
}

// === Writer impl ===
//
// Delegates all methods to `self.inner`. `bit()` is the Tier 1
// L(1) intercept site: checks `self.plan` for an override at
// `self.cursor` and substitutes the override value if present;
// always advances the cursor.

impl Writer for WriterStego {
    fn symbol<const CDF_LEN: usize>(&mut self, s: u32, cdf: &[u16; CDF_LEN]) {
        self.inner.symbol(s, cdf);
    }

    fn symbol_bits(&self, s: u32, cdf: &[u16]) -> u32 {
        self.inner.symbol_bits(s, cdf)
    }

    fn symbol_with_update<const CDF_LEN: usize>(
        &mut self,
        s: u32,
        cdf: CDFOffset<CDF_LEN>,
        log: &mut CDFContextLog,
        fc: &mut CDFContext,
    ) {
        self.inner.symbol_with_update(s, cdf, log, fc);
    }

    fn bool(&mut self, val: bool, f: u16) {
        // W3.6: 50/50 bool() calls are L(1) candidates equivalent to
        // bit() (rav1e routes `bit(b)` through `bool(b==1, 16384)`).
        // Non-bit() direct callers like `block_unit.rs:1770` delta_lf
        // sign + `ec.rs::667` write_subexp flag emit through bool()
        // directly. Mirror the fork's phasm_track_bit semantics here:
        // intercept on f==16384, apply override at the current cursor,
        // advance cursor. This keeps WriterStego encode-time + hybrid
        // replay byte-identical (verified by
        // `hybrid_replay_matches_writer_stego_encode_time_override`
        // and `hybrid_replay_catches_direct_bool_16384_emission`).
        if f == 16384 {
            let effective = match self.plan.get(self.cursor) {
                Some(override_bit) => override_bit != 0,
                None => val,
            };
            self.cursor = self.cursor.wrapping_add(1);
            self.inner.bool(effective, f);
        } else {
            self.inner.bool(val, f);
        }
    }

    fn bit(&mut self, bit: u16) {
        // Tier 1 L(1) intercept: apply override if planned at this
        // cursor position; advance cursor unconditionally.
        let effective = match self.plan.get(self.cursor) {
            Some(override_bit) => override_bit,
            None => bit,
        };
        self.inner.bit(effective);
        self.cursor = self.cursor.wrapping_add(1);
    }

    fn literal(&mut self, bits: u8, s: u32) {
        // L(n) literal: each bit goes through `bit()` which has the
        // override + cursor advance. Need to mirror rav1e's bit-order:
        // rav1e's WriterBase emits literal high-bit-first (i.e., bit
        // `bits-1`, ..., bit 0). Verified by reading WriterBase::literal
        // in src/ec.rs.
        for i in (0..bits).rev() {
            let b = ((s >> i) & 1) as u16;
            self.bit(b);
        }
    }

    fn write_golomb(&mut self, level: u32) {
        // W3.4: re-implements rav1e's write_golomb (src/ec.rs::609)
        // routing every emitted bit through WriterStego::bit so the
        // cursor advances AND the Tier 1 override plan can intercept
        // golomb data bits (channel-design.md § 4.2 `GolombTailLsb`).
        //
        // Identical bit pattern to inner.write_golomb(level): emit
        // (length-1) prefix zero-bits, then `length` data bits
        // MSB-first.
        let x = level + 1;
        let length = 32 - x.leading_zeros();
        for _ in 0..length - 1 {
            self.bit(0);
        }
        for i in (0..length).rev() {
            self.bit(((x >> i) & 0x01) as u16);
        }
    }

    fn write_quniform(&mut self, n: u32, v: u32) {
        self.inner.write_quniform(n, v);
    }

    fn count_quniform(&self, n: u32, v: u32) -> u32 {
        self.inner.count_quniform(n, v)
    }

    fn write_subexp(&mut self, n: u32, k: u8, v: u32) {
        self.inner.write_subexp(n, k, v);
    }

    fn count_subexp(&self, n: u32, k: u8, v: u32) -> u32 {
        self.inner.count_subexp(n, k, v)
    }

    fn write_unsigned_subexp_with_ref(&mut self, v: u32, mx: u32, k: u8, r: u32) {
        self.inner.write_unsigned_subexp_with_ref(v, mx, k, r);
    }

    fn count_unsigned_subexp_with_ref(
        &self,
        v: u32,
        mx: u32,
        k: u8,
        r: u32,
    ) -> u32 {
        self.inner.count_unsigned_subexp_with_ref(v, mx, k, r)
    }

    fn write_signed_subexp_with_ref(
        &mut self,
        v: i32,
        low: i32,
        high: i32,
        k: u8,
        r: i32,
    ) {
        self.inner.write_signed_subexp_with_ref(v, low, high, k, r);
    }

    fn count_signed_subexp_with_ref(
        &self,
        v: i32,
        low: i32,
        high: i32,
        k: u8,
        r: i32,
    ) -> u32 {
        self.inner.count_signed_subexp_with_ref(v, low, high, k, r)
    }

    fn tell(&mut self) -> u32 {
        <WriterBase<WriterEncoder> as Writer>::tell(&mut self.inner)
    }

    fn tell_frac(&mut self) -> u32 {
        <WriterBase<WriterEncoder> as Writer>::tell_frac(&mut self.inner)
    }

    fn checkpoint(&mut self) -> WriterCheckpoint {
        <WriterBase<WriterEncoder> as Writer>::checkpoint(&mut self.inner)
    }

    fn rollback(&mut self, cp: &WriterCheckpoint) {
        <WriterBase<WriterEncoder> as Writer>::rollback(&mut self.inner, cp);
    }

    fn add_bits_frac(&mut self, bits_frac: u32) {
        self.inner.add_bits_frac(bits_frac);
    }
}

// === StorageBackend impl ===
//
// Required so `WriterRecorder::replay(&mut dyn StorageBackend)`
// can stream cached symbols through WriterStego during Pass 2.
// All four methods delegate to the inner `WriterBase<WriterEncoder>`
// via trait disambiguation (Writer + StorageBackend both define
// checkpoint/rollback).

impl StorageBackend for WriterStego {
    fn store(&mut self, fl: u16, fh: u16, nms: u16) {
        // TODO W3.4: store-level intercept site for cases where
        // override is needed below the bit() layer. For Tier 1 L(1)
        // overrides the bit() intercept already fired before this
        // method is reached, so no extra work needed here for v0.3.
        self.inner.store(fl, fh, nms);
    }

    fn stream_bits(&mut self) -> usize {
        <WriterBase<WriterEncoder> as StorageBackend>::stream_bits(&mut self.inner)
    }

    fn checkpoint(&mut self) -> WriterCheckpoint {
        <WriterBase<WriterEncoder> as StorageBackend>::checkpoint(&mut self.inner)
    }

    fn rollback(&mut self, cp: &WriterCheckpoint) {
        <WriterBase<WriterEncoder> as StorageBackend>::rollback(&mut self.inner, cp);
    }
}

// === Encoder-side cover positions (W3.10.1, collapsed-W3.9 helper) ===
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
// for every N — verified by `w3d43_*` + W3.10.2 tests in
// `decoder.rs::round_trip`.

/// A single L(1) cover position from a recorded encoder pass. Cursor
/// is the monotonic 50/50 emission index starting at 0; matches the
/// OverrideMap key space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoverPosition {
    /// Monotonic L(1) emission count — matches the OverrideMap key
    /// AND the decoder-side cursor (W3.10.2 parity invariant).
    pub cursor: u64,
    /// Index into `recorder.phasm_storage()` where the matching
    /// `(fl, fh, nms)` tuple lives. Used by `replay_with_overrides`.
    pub storage_index: u32,
    /// The natural bit value at this position (0 or 1) — what the
    /// encoder would emit absent any stego override.
    pub natural_value: u16,
    /// Channel identity per W3.9.0 tagging:
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
// Pairs phasm-rav1e's W3.6 fork patch with phasm-core's STC override
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
/// Derived from rav1e `src/ec.rs::485-489` + `src/ec.rs::520-537`
/// (W3.5 source-read):
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
    fn skeleton_constructs() {
        let stego = WriterStego::new();
        assert_eq!(stego.cursor(), 0);
    }

    #[test]
    fn default_is_empty() {
        let stego = WriterStego::default();
        assert_eq!(stego.cursor(), 0);
        assert!(stego.plan.is_empty());
    }

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
    fn writer_bit_advances_cursor() {
        let mut stego = WriterStego::new();
        assert_eq!(stego.cursor(), 0);
        stego.bit(1);
        assert_eq!(stego.cursor(), 1);
        stego.bit(0);
        assert_eq!(stego.cursor(), 2);
    }

    #[test]
    fn writer_literal_advances_cursor_by_nbits() {
        let mut stego = WriterStego::new();
        stego.literal(8, 0xab);
        assert_eq!(stego.cursor(), 8);
        stego.literal(4, 0x5);
        assert_eq!(stego.cursor(), 12);
    }

    #[test]
    fn no_overrides_unchanged_path() {
        // With empty plan, every bit() call delegates the natural
        // value to the inner WriterEncoder. Compile-time check that
        // cursor advances without override interference.
        let mut stego = WriterStego::new();
        for i in 0..32 {
            stego.bit((i & 1) as u16);
        }
        assert_eq!(stego.cursor(), 32);
        assert!(stego.plan.is_empty());
    }

    #[test]
    fn rav1e_ec_types_importable() {
        // Verify the rav1e type imports still resolve (W3.2 carry-
        // forward; ensures fork submodule + Cargo dep wiring stays
        // healthy after the W3.3 expansion).
        fn _accept_writer<W: Writer>(_: &mut W) {}
        fn _accept_storage<S: StorageBackend>(_: &mut S) {}
        let mut stego = WriterStego::new();
        _accept_writer(&mut stego);
        _accept_storage(&mut stego);
    }

    // === W3.4 additions ===

    #[test]
    fn finalize_returns_bytes() {
        // Writing a handful of bits then finalizing produces a
        // non-empty bitstream. Validates the `done()` wiring.
        let mut stego = WriterStego::new();
        for i in 0..16 {
            stego.bit((i & 1) as u16);
        }
        let bytes = stego.finalize();
        // Range coder produces at least a few bytes for 16 fair-coin
        // bits.
        assert!(!bytes.is_empty(), "finalize() should produce bytes for 16-bit input");
    }

    #[test]
    fn finalize_empty_produces_some_state() {
        // Even with zero symbols written, the range coder's
        // finalization emits the trailing termination bits. Just
        // check it doesn't panic.
        let stego = WriterStego::new();
        let _ = stego.finalize();
    }

    #[test]
    fn write_golomb_advances_cursor() {
        // write_golomb(level=0): x=1, length=1, prefix 0 bits, 1 data bit → cursor += 1
        // write_golomb(level=1): x=2, length=2, prefix 1 bit, 2 data bits → cursor += 3
        // write_golomb(level=3): x=4, length=3, prefix 2 bits, 3 data bits → cursor += 5
        let mut stego = WriterStego::new();
        stego.write_golomb(0);
        assert_eq!(stego.cursor(), 1, "golomb(0) emits 1 bit");
        stego.write_golomb(1);
        assert_eq!(stego.cursor(), 1 + 3, "golomb(1) emits 3 bits");
        stego.write_golomb(3);
        assert_eq!(stego.cursor(), 1 + 3 + 5, "golomb(3) emits 5 bits");
    }

    #[test]
    fn override_map_hashmap_o1_lookup() {
        // After W3.4's Vec → HashMap switch, lookup is O(1) instead
        // of O(N). Sanity-check that get() still returns the right
        // value when many entries are present.
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
        // HashMap semantics: set() with same key overwrites.
        let mut plan = OverrideMap::new();
        plan.set(42, 0);
        plan.set(42, 1);
        assert_eq!(plan.get(42), Some(1));
        assert_eq!(plan.len(), 1);
    }

    #[test]
    fn override_applies_at_planned_position() {
        // Plan an override at position 3. Write 8 bits via literal();
        // the 4th bit (cursor==3 at that moment) should be substituted.
        // This test verifies the cursor advancement + override lookup
        // at a deterministic position. We can't directly observe which
        // bit got emitted (would require decoding the final bytes),
        // but we can verify cursor advanced correctly + the plan was
        // queried (via behavioural observation).
        let mut plan = OverrideMap::new();
        plan.set(3, 1);
        let mut stego = WriterStego::new();
        stego.set_plan(plan);
        stego.literal(8, 0b00000000);
        assert_eq!(stego.cursor(), 8, "literal(8) advances cursor by 8");
        // The override at position 3 substituted bit value 1 for the
        // natural 0; round-trip verification deferred to W3.5+ when
        // we have a decode-side walker.
    }

    #[test]
    fn override_applies_in_golomb_data_bits() {
        // W3.4 key feature: write_golomb now routes through
        // WriterStego::bit (not inner.write_golomb), so overrides at
        // golomb data-bit positions take effect — closes the
        // GolombTailLsb cursor-tracking gap noted in W3.3.
        let mut plan = OverrideMap::new();
        plan.set(2, 1); // override the 3rd bit emitted by golomb(1)
        let mut stego = WriterStego::new();
        stego.set_plan(plan);
        stego.write_golomb(1); // emits 3 bits: prefix '0', data '1', data '0'
        assert_eq!(stego.cursor(), 3);
        // Position 2 (the 3rd bit = data '0') was planned to override
        // to '1'. Cursor-level verification only; bitstream-level
        // verification covered by W3.5 tests below.
    }

    // === W3.5: Pass 2 cached-replay round-trip tests ===
    //
    // These tests exercise the full WriterStego ↔ WriterBase<WriterEncoder>
    // ↔ WriterRecorder pipeline against the streaming-session.md § 1.1
    // Option E architecture. Two architectural shapes are tested:
    //
    // A. WriterRecorder replay-through-StorageBackend byte-identity:
    //    WriterStego's StorageBackend impl correctly forwards to the
    //    inner WriterBase<WriterEncoder>, so replaying the same
    //    recorded symbols through either path produces identical bytes.
    //
    // B. Tier 1 emit-time override changes bitstream bytes:
    //    Two WriterStego instances fed identical Writer-trait inputs,
    //    one with an active override plan, produce DIFFERENT output
    //    bytes — proves the override actually took effect on the wire.
    //
    // Architectural note: rav1e's WriterRecorder.replay() routes
    // through StorageBackend::store(), NOT through Writer::bit(). So
    // Tier 1 overrides at Writer::bit() level do NOT fire during
    // replay — they only fire during direct Writer-trait writes.
    // That's the audit-confirmed semantics; W3.5+ Pass 2 architecture
    // therefore re-encodes (rather than replays) for cases where
    // overrides need to apply. Verified by Q11 spike
    // (research/16-spike-Q11-rav1e-api.md): rav1e is deterministic,
    // so re-encoding the same input twice produces byte-identical
    // output.

    use phasm_rav1e::ec::WriterRecorder as RavWriterRecorder;

    /// W3.5 Test A — replay forwarding byte-identity.
    ///
    /// Replay the same WriterRecorder through:
    ///   sink_a = WriterBase<WriterEncoder> (rav1e native sink)
    ///   sink_b = WriterStego with empty plan
    /// Both must produce byte-identical output. Proves WriterStego's
    /// StorageBackend impl correctly forwards to the inner without
    /// silently corrupting range-coder state.
    #[test]
    fn pass2_replay_byte_identity_against_native_sink() {
        // Record one set of bits via WriterRecorder.
        let mut recorder = RavWriterRecorder::new();
        for i in 0..16 {
            recorder.bit((i & 1) as u16);
        }
        recorder.literal(8, 0xab);

        // Sink A: native WriterBase<WriterEncoder>.
        let mut sink_a = WriterEncoder::new();
        recorder.replay(&mut sink_a);
        let bytes_a = sink_a.done();

        // Sink B: WriterStego with empty plan. Re-record (since
        // replay consumes the recorder's state in some
        // implementations; safer to use a fresh recorder for the
        // second run).
        let mut recorder2 = RavWriterRecorder::new();
        for i in 0..16 {
            recorder2.bit((i & 1) as u16);
        }
        recorder2.literal(8, 0xab);

        let mut sink_b = WriterStego::new();
        recorder2.replay(&mut sink_b);
        let bytes_b = sink_b.finalize();

        assert_eq!(
            bytes_a, bytes_b,
            "WriterStego StorageBackend impl must forward to inner without altering bytes"
        );
    }

    /// W3.5 Test B — direct writing byte-identity (no replay).
    ///
    /// Write identical inputs through WriterBase<WriterEncoder>
    /// directly and through WriterStego with empty plan; bytes must
    /// be byte-identical. Proves WriterStego's Writer-trait impl
    /// also forwards to inner correctly.
    #[test]
    fn empty_plan_writer_byte_identity_against_native() {
        let mut native = WriterEncoder::new();
        for i in 0..16u32 {
            native.bit((i & 1) as u16);
        }
        native.literal(8, 0xab);
        native.write_golomb(3);
        let bytes_native = native.done();

        let mut stego = WriterStego::new();
        for i in 0..16u32 {
            stego.bit((i & 1) as u16);
        }
        stego.literal(8, 0xab);
        stego.write_golomb(3);
        let bytes_stego = stego.finalize();

        assert_eq!(
            bytes_native, bytes_stego,
            "WriterStego with empty plan must produce byte-identical output to WriterBase<WriterEncoder>"
        );
    }

    /// W3.5 Test C — Tier 1 override changes bitstream bytes.
    ///
    /// Two WriterStego instances fed identical inputs, one with an
    /// active override plan: the bytes MUST differ — proves the
    /// override actually took effect on the wire (not just at the
    /// cursor-tracking level).
    #[test]
    fn override_changes_output_bytes() {
        // literal(8, 0xab) emits bits MSB-first at cursor positions
        // 0..7. 0xab = 0b10101011, so the bit at cursor=3 is 0
        // (the 4th-from-MSB bit). Plan an override flipping it to 1.
        let mut stego_clean = WriterStego::new();
        stego_clean.literal(8, 0xab);
        let bytes_clean = stego_clean.finalize();

        let mut plan = OverrideMap::new();
        plan.set(3, 1); // flip position 3 from 0 → 1

        let mut stego_dirty = WriterStego::new();
        stego_dirty.set_plan(plan);
        stego_dirty.literal(8, 0xab);
        let bytes_dirty = stego_dirty.finalize();

        assert_ne!(
            bytes_clean, bytes_dirty,
            "override at cursor=3 must change the output bitstream"
        );
    }

    /// W3.5 Test D — override at a position equal to the natural bit
    /// must NOT change output (no-op override).
    #[test]
    fn override_matching_natural_value_is_noop() {
        // 0xab = 0b10101011. Bit at cursor=0 is the MSB = 1.
        // Overriding cursor=0 to 1 (same as natural) should be a no-op.
        let mut stego_clean = WriterStego::new();
        stego_clean.literal(8, 0xab);
        let bytes_clean = stego_clean.finalize();

        let mut plan = OverrideMap::new();
        plan.set(0, 1); // natural is already 1

        let mut stego_overridden = WriterStego::new();
        stego_overridden.set_plan(plan);
        stego_overridden.literal(8, 0xab);
        let bytes_overridden = stego_overridden.finalize();

        assert_eq!(
            bytes_clean, bytes_overridden,
            "override matching the natural bit value must produce identical bytes"
        );
    }

    /// W3.5 Test E — multiple overrides accumulate (each takes effect).
    #[test]
    fn multiple_overrides_accumulate() {
        // 0xab = 0b10101011 → natural bits MSB-first: 1,0,1,0,1,0,1,1.
        // Flip every even-indexed natural bit to its inverse.
        let mut stego_clean = WriterStego::new();
        stego_clean.literal(8, 0xab);
        let bytes_clean = stego_clean.finalize();

        let mut plan = OverrideMap::new();
        plan.set(0, 0); // 1 → 0
        plan.set(2, 0); // 1 → 0
        plan.set(4, 0); // 1 → 0
        plan.set(6, 0); // 1 → 0

        let mut stego_overridden = WriterStego::new();
        stego_overridden.set_plan(plan);
        stego_overridden.literal(8, 0xab);
        let bytes_overridden = stego_overridden.finalize();

        // With 4 of 8 bits flipped, the byte stream must differ.
        assert_ne!(
            bytes_clean, bytes_overridden,
            "4 overrides must change the output bitstream"
        );
    }

    /// W3.5 Test F — empty plan = byte-identical to direct WriterEncoder
    /// across a longer sample (regression guard for any subtle Writer
    /// trait method forwarding bug).
    #[test]
    fn empty_plan_byte_identity_long_sample() {
        // Mix of bit / literal / golomb across many cursor positions.
        fn write_sample<W: Writer>(w: &mut W) {
            for i in 0..50u32 {
                w.bit((i.wrapping_mul(7) & 1) as u16);
            }
            w.literal(8, 0xab);
            w.literal(8, 0xcd);
            w.literal(4, 0x5);
            for level in 0..5u32 {
                w.write_golomb(level);
            }
            w.bool(true, 16384);
            w.bool(false, 8192);
        }

        let mut native = WriterEncoder::new();
        write_sample(&mut native);
        let bytes_native = native.done();

        let mut stego = WriterStego::new();
        write_sample(&mut stego);
        let bytes_stego = stego.finalize();

        assert_eq!(
            bytes_native, bytes_stego,
            "WriterStego with empty plan must be byte-identical to native across a varied input"
        );
    }

    // === W3.6: hybrid Pass 2 replay-with-overrides ===
    //
    // These tests exercise the phasm-rav1e W3.6 fork patch + the
    // phasm-core `replay_with_overrides` function. The patch makes
    // `WriterRecorder` track 50/50 L(1) emission positions
    // (`phasm_bit_positions`) parallel to its `storage` Vec, enabling
    // bit-level override at replay time without the full re-encode
    // Pass 2 cost that the W3.5 finding had implied.
    //
    // Coverage:
    //  - empty plan → byte-identical to direct WriterEncoder
    //  - empty plan → byte-identical to WriterStego encode-time path
    //  - override changes bytes
    //  - override matches WriterStego encode-time override byte-exact
    //  - override on write_golomb data bits
    //  - override on direct bool(_, 16384) call (delta_lf-style sign)
    //  - long varied sample byte-identity (regression guard)
    //  - rollback truncates phasm_bit_positions correctly

    /// W3.6-A: hybrid replay with empty plan is byte-identical to a
    /// direct `WriterBase<WriterEncoder>` encode of the same input.
    #[test]
    fn hybrid_replay_empty_plan_byte_identity() {
        fn write_sample<W: Writer>(w: &mut W) {
            for i in 0..16u32 {
                w.bit((i & 1) as u16);
            }
            w.literal(8, 0xab);
            w.write_golomb(3);
        }

        let mut native = WriterEncoder::new();
        write_sample(&mut native);
        let bytes_native = native.done();

        let mut recorder = WriterRecorder::new();
        write_sample(&mut recorder);
        let bytes_replay = finalize_replay(&recorder, &OverrideMap::new());

        assert_eq!(
            bytes_native, bytes_replay,
            "empty-plan hybrid replay must be byte-identical to a direct encode"
        );
    }

    /// W3.6-B: hybrid replay with override changes bytes (override
    /// actually fires).
    #[test]
    fn hybrid_replay_override_changes_bytes() {
        let mut recorder = WriterRecorder::new();
        recorder.literal(8, 0xab);

        let bytes_clean = finalize_replay(&recorder, &OverrideMap::new());

        let mut plan = OverrideMap::new();
        plan.set(3, 1); // 0xab MSB-first: 1,0,1,0,1,0,1,1; flip bit[3]=0→1
        let bytes_dirty = finalize_replay(&recorder, &plan);

        assert_ne!(
            bytes_clean, bytes_dirty,
            "hybrid replay must change bytes when the plan flips a bit"
        );
    }

    /// W3.6-C: hybrid replay byte-matches the encode-time
    /// `WriterStego::bit` override path. This is the load-bearing
    /// equivalence: it proves Pass 2 hybrid replay IS THE SAME as a
    /// Pass-2 full re-encode through WriterStego, so we can use the
    /// cheap replay path instead of paying for re-encode.
    #[test]
    fn hybrid_replay_matches_writer_stego_encode_time_override() {
        fn write_sample<W: Writer>(w: &mut W) {
            w.literal(8, 0xab);
            w.write_golomb(3);
            w.bit(0);
            w.bit(1);
            w.literal(4, 0x5);
        }

        let mut plan = OverrideMap::new();
        plan.set(2, 1);
        plan.set(5, 0);
        plan.set(11, 1);

        // Encode-time path: WriterStego applies overrides during the
        // initial encode (Path 1c re-encode equivalent).
        let mut stego = WriterStego::new();
        stego.set_plan(plan.clone());
        write_sample(&mut stego);
        let bytes_encode_time = stego.finalize();

        // Replay path: WriterRecorder captures, then hybrid replay
        // applies the same plan.
        let mut recorder = WriterRecorder::new();
        write_sample(&mut recorder);
        let bytes_replay = finalize_replay(&recorder, &plan);

        assert_eq!(
            bytes_encode_time, bytes_replay,
            "hybrid replay must produce byte-identical output to a re-encode through WriterStego"
        );
    }

    /// W3.6-D: hybrid replay override fires on `write_golomb` data
    /// bits (the GolombTailLsb Tier 1 secondary channel per
    /// channel-design.md § 4.2). Verified by override-vs-no-override
    /// byte difference AND by equivalence with the WriterStego
    /// encode-time path.
    #[test]
    fn hybrid_replay_golomb_data_bit_override() {
        // write_golomb(7): x=8, length=4, prefix '000' (length-1=3
        // zeros at cursors 0..2), then `length`=4 data bits MSB-first
        // of x=0b1000 → cursor 3='1', cursor 4='0', cursor 5='0',
        // cursor 6='0'. Total 7 bits on the wire.
        // The leading data '1' at cursor=3 is the only non-zero bit;
        // flipping it to 0 must change the bytes vs empty plan.
        let mut plan = OverrideMap::new();
        plan.set(3, 0);

        let mut stego = WriterStego::new();
        stego.set_plan(plan.clone());
        stego.write_golomb(7);
        let bytes_encode = stego.finalize();

        let mut recorder = WriterRecorder::new();
        recorder.write_golomb(7);
        let bytes_replay = finalize_replay(&recorder, &plan);

        assert_eq!(
            bytes_encode, bytes_replay,
            "golomb data-bit override must match between encode-time and replay paths"
        );

        let bytes_clean = finalize_replay(&recorder, &OverrideMap::new());
        assert_ne!(
            bytes_clean, bytes_replay,
            "golomb data-bit override must change bytes vs empty plan"
        );
    }

    /// W3.6-E: hybrid replay catches direct `bool(_, 16384)` calls
    /// (the rav1e pattern at `src/context/block_unit.rs:1770` for
    /// delta_lf sign + `src/ec.rs::667` inside write_subexp). These
    /// emissions do NOT route through `bit()` but are still 50/50
    /// L(1) candidates; the fork patch's bool() hook catches them.
    #[test]
    fn hybrid_replay_catches_direct_bool_16384_emission() {
        // Write a `bool(true, 16384)` directly (NOT through bit()) —
        // this is the rav1e pattern used at block_unit.rs:1770 for
        // delta_lf sign emission and at ec.rs:667 inside write_subexp.
        fn write_sample<W: Writer>(w: &mut W) {
            w.bool(true, 16384);
            w.bool(false, 16384);
            w.bool(true, 16384);
        }

        let mut plan = OverrideMap::new();
        plan.set(1, 1); // flip middle one: false → true

        let mut stego = WriterStego::new();
        stego.set_plan(plan.clone());
        write_sample(&mut stego);
        let bytes_encode = stego.finalize();

        let mut recorder = WriterRecorder::new();
        write_sample(&mut recorder);
        let bytes_replay = finalize_replay(&recorder, &plan);

        // Both paths must produce identical output, AND the fork hook
        // must have caught all three bool(_, 16384) calls (verifiable
        // via the recorder's bit_positions field — see next test).
        assert_eq!(
            bytes_encode, bytes_replay,
            "direct bool(_, 16384) emissions must be hooked at replay parity with encode-time"
        );
    }

    /// W3.6-F: verify the fork's `phasm_bit_positions` index sees
    /// exactly the expected count of 50/50 emissions across mixed
    /// Writer methods.
    #[test]
    fn fork_hook_counts_all_50_50_emissions() {
        let mut recorder = WriterRecorder::new();
        recorder.bit(0); // +1
        recorder.bit(1); // +1
        recorder.literal(8, 0xab); // +8
        recorder.write_golomb(3); // x=4, length=3 → 2 prefix + 3 data = +5
        recorder.bool(true, 16384); // +1
        recorder.bool(false, 16384); // +1
        recorder.bool(true, 8192); // NOT 50/50 → +0
        recorder.bool(false, 24576); // NOT 50/50 → +0

        let positions = recorder.phasm_bit_positions();
        assert_eq!(
            positions.len(),
            1 + 1 + 8 + 5 + 1 + 1,
            "hook should fire for every 50/50 (f==16384) bool() call"
        );

        // Verify natural bit values are captured (cursor 0=bit(0)=0,
        // cursor 1=bit(1)=1, then 0xab MSB bits: 1,0,1,0,1,0,1,1).
        assert_eq!(positions[0].1, 0); // bit(0)
        assert_eq!(positions[1].1, 1); // bit(1)
        assert_eq!(positions[2].1, 1); // 0xab MSB
        assert_eq!(positions[3].1, 0);
        assert_eq!(positions[4].1, 1);
        assert_eq!(positions[5].1, 0);
        assert_eq!(positions[6].1, 1);
        assert_eq!(positions[7].1, 0);
        assert_eq!(positions[8].1, 1);
        assert_eq!(positions[9].1, 1); // 0xab LSB
    }

    /// W3.6-G: long varied sample byte-identity (regression guard).
    /// Mix of bit, literal, write_golomb, bool(_, 16384), and a
    /// non-50/50 bool(_, 8192) — all replayed must match a direct
    /// encode with the same input.
    #[test]
    fn hybrid_replay_long_sample_byte_identity() {
        fn write_sample<W: Writer>(w: &mut W) {
            for i in 0..50u32 {
                w.bit((i.wrapping_mul(7) & 1) as u16);
            }
            w.literal(8, 0xab);
            w.literal(8, 0xcd);
            w.literal(4, 0x5);
            for level in 0..5u32 {
                w.write_golomb(level);
            }
            w.bool(true, 16384);
            w.bool(false, 16384);
            // Non-50/50: fork hook should NOT fire here.
            w.bool(false, 8192);
        }

        let mut native = WriterEncoder::new();
        write_sample(&mut native);
        let bytes_native = native.done();

        let mut recorder = WriterRecorder::new();
        write_sample(&mut recorder);
        let bytes_replay = finalize_replay(&recorder, &OverrideMap::new());

        assert_eq!(
            bytes_native, bytes_replay,
            "long varied sample: empty-plan hybrid replay must match a direct encode"
        );
    }

    // === W3.10.1: CoverPositions wrapper tests ===

    #[test]
    fn cover_positions_from_empty_recorder() {
        let positions = CoverPositions::from_recorder(&[], &[]);
        assert!(positions.is_empty());
        assert_eq!(positions.len(), 0);
        assert_eq!(positions.count_by_tag(0), 0);
    }

    #[test]
    fn cover_positions_mismatched_lengths_panics() {
        let result = std::panic::catch_unwind(|| {
            CoverPositions::from_recorder(&[(0, 1)], &[])
        });
        assert!(
            result.is_err(),
            "from_recorder should panic on mismatched array lengths"
        );
    }

    #[test]
    fn cover_positions_synthetic_construction() {
        use phasm_rav1e::phasm_stego::{
            PHASM_TAG_AC_COEFF_SIGN, PHASM_TAG_GOLOMB_TAIL_LSB, PHASM_TAG_OTHER,
        };
        let bit_positions = [(0u32, 1u16), (5, 0), (10, 1), (15, 0)];
        let bit_tags = [
            PHASM_TAG_OTHER,
            PHASM_TAG_AC_COEFF_SIGN,
            PHASM_TAG_AC_COEFF_SIGN,
            PHASM_TAG_GOLOMB_TAIL_LSB,
        ];
        let positions = CoverPositions::from_recorder(&bit_positions, &bit_tags);

        assert_eq!(positions.len(), 4);
        assert_eq!(positions.count_by_tag(PHASM_TAG_OTHER), 1);
        assert_eq!(positions.count_by_tag(PHASM_TAG_AC_COEFF_SIGN), 2);
        assert_eq!(positions.count_by_tag(PHASM_TAG_GOLOMB_TAIL_LSB), 1);

        // Verify cursor monotonicity + content.
        let collected: Vec<_> = positions.iter().collect();
        assert_eq!(collected[0].cursor, 0);
        assert_eq!(collected[0].storage_index, 0);
        assert_eq!(collected[0].natural_value, 1);
        assert_eq!(collected[0].tag, PHASM_TAG_OTHER);

        assert_eq!(collected[1].cursor, 1);
        assert_eq!(collected[1].storage_index, 5);
        assert_eq!(collected[1].tag, PHASM_TAG_AC_COEFF_SIGN);

        // filter_by_tag preserves cursor order.
        let ac_signs = positions.filter_by_tag(PHASM_TAG_AC_COEFF_SIGN);
        assert_eq!(ac_signs.len(), 2);
        assert_eq!(ac_signs[0].cursor, 1);
        assert_eq!(ac_signs[1].cursor, 2);
    }

    /// W3.6-H: rollback truncates `phasm_bit_positions` correctly.
    /// Verify the fork's rollback hook keeps the index in sync with
    /// the truncated storage (so subsequent replays don't reference
    /// dropped positions).
    #[test]
    fn fork_rollback_truncates_bit_positions() {
        let mut recorder = WriterRecorder::new();
        recorder.bit(0);
        recorder.bit(1);
        let checkpoint = <WriterBase<WriterRecorder> as Writer>::checkpoint(&mut recorder);
        let pre_count = recorder.phasm_bit_positions().len();
        assert_eq!(pre_count, 2, "should have 2 bit positions before checkpoint");

        recorder.bit(0);
        recorder.literal(4, 0x5); // +4 bit positions
        assert_eq!(recorder.phasm_bit_positions().len(), 2 + 1 + 4);

        <WriterBase<WriterRecorder> as Writer>::rollback(&mut recorder, &checkpoint);
        assert_eq!(
            recorder.phasm_bit_positions().len(),
            pre_count,
            "rollback must truncate phasm_bit_positions to checkpoint state"
        );
    }

    // === W3.8.6: end-to-end Pass 1 + Pass 2 round-trip via rav1e fork ===
    //
    // Closes the W3.8 loop by combining the rav1e fork's Option 3
    // surface (encode_tile::<WriterRecorder> + re-exported internal
    // types per rav1e-hook-sites.md § 3.0) with phasm-core's
    // hybrid-replay machinery (replay_with_overrides + WriterStego).
    //
    // Three assertions:
    //   1. Round-trip byte-identity: replay_with_overrides on the
    //      recorder data, EMPTY plan → bytes byte-identical to a
    //      direct encode_tile::<WriterEncoder> pass.
    //   2. Override fires on the wire: same recorder, NON-empty plan
    //      → bytes differ from natural.
    //   3. Length invariant: stego bytes have the SAME LENGTH as
    //      natural bytes (Tier 1 50/50 flips consume the same bit
    //      count regardless of value — load-bearing for the OBU-splice
    //      strategy in rav1e-hook-sites.md § 3.1).
    //
    // This is the first test that drives Pass 1 through real rav1e
    // encode logic (not synthetic WriterRecorder input). Validates
    // the entire Option 3 architecture end-to-end before W3.9
    // (cover-position walker) + W3.10 (orchestrator + STC plan)
    // land on top of it.

    use phasm_rav1e::phasm_stego::{
        encode_tile, make_frame, make_inter_config, FrameInvariants, FrameState,
        InterConfig,
    };
    use phasm_rav1e::prelude::Sequence;
    use phasm_rav1e::{EncoderConfig, Frame, Pixel};
    use std::sync::Arc;

    /// Construct a tiny 64×64 single-tile key-frame state with a
    /// gradient-filled input. Mirrors `phasm-rav1e/src/encoder.rs`
    /// `phasm_smoke_tests::setup_frame_state()` but drives via the
    /// public `phasm_stego` re-export module so we exercise the
    /// Option 3 API surface that phasm-core's W3.10 orchestrator
    /// will eventually depend on.
    fn setup_rav1e_state() -> (
        FrameInvariants<u8>,
        FrameState<u8>,
        InterConfig,
    ) {
        use phasm_rav1e::color::ChromaSampling;
        let config = Arc::new(EncoderConfig {
            width: 64,
            height: 64,
            bit_depth: 8,
            chroma_sampling: ChromaSampling::Cs420,
            ..Default::default()
        });
        let mut sequence = Sequence::new(&config);
        sequence.enable_large_lru = false;
        let fi = FrameInvariants::<u8>::new_key_frame(
            config.clone(),
            Arc::new(sequence),
            0,
            Box::new([]),
        );
        let mut frame = make_frame::<u8>(
            fi.width,
            fi.height,
            fi.sequence.chroma_sampling,
        );
        fill_gradient(&mut frame);
        let fs = FrameState::new_with_frame(&fi, Arc::new(frame));
        let inter_cfg = make_inter_config(&config);
        (fi, fs, inter_cfg)
    }

    /// Deterministic gradient — required for the encoder to produce
    /// non-zero AC coefficients, which is what creates L(1) emission
    /// sites (per phasm-rav1e smoke-test learning, captured in
    /// rav1e-hook-sites.md § 3.4 DEF-2).
    fn fill_gradient<T: Pixel>(frame: &mut Frame<T>) {
        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let stride = plane.cfg.stride;
            for (row_idx, row) in plane.data.chunks_mut(stride).enumerate() {
                for (col_idx, pixel) in row.iter_mut().enumerate() {
                    let val =
                        ((row_idx.wrapping_mul(7) + col_idx.wrapping_mul(3) + plane_idx * 13)
                            & 0xff) as u8;
                    *pixel = T::cast_from(val);
                }
            }
        }
    }

    /// W3.8.6 Test A — empty-plan hybrid replay is byte-identical to
    /// a direct rav1e natural encode via encode_tile::<WriterEncoder>.
    ///
    /// This is the load-bearing equivalence for the entire Option E
    /// architecture: if it fails, the W3.6 fork hook is broken OR
    /// rav1e's encoder isn't deterministic byte-by-byte across two
    /// invocations.
    #[test]
    fn w386_round_trip_empty_plan_byte_identity() {
        use phasm_rav1e::context::CDFContext;
        use phasm_rav1e::phasm_stego::FrameBlocks;
        use phasm_rav1e::ec::WriterEncoder as NativeEnc;

        // Pass 1: encode with WriterRecorder, capture storage + bit_positions.
        let (fi, mut fs, inter_cfg) = setup_rav1e_state();
        let mut blocks = FrameBlocks::new(fi.w_in_b, fi.h_in_b);
        let mut cdf_recording = CDFContext::new(fi.base_q_idx);
        let ti = &fi.sequence.tiling;
        let recorder = {
            let mut iter = ti.tile_iter_mut(&mut fs, &mut blocks);
            let mut ctx = iter
                .next()
                .expect("single-tile config yields one tile");
            drop(iter);
            let (recorder, _stats): (WriterBase<WriterRecorder>, _) =
                encode_tile(&fi, &mut ctx.ts, &mut cdf_recording, &mut ctx.tb, &inter_cfg);
            recorder
        };

        // Natural encode: re-run with WriterEncoder to get bytes for comparison.
        let (fi2, mut fs2, inter_cfg2) = setup_rav1e_state();
        let mut blocks2 = FrameBlocks::new(fi2.w_in_b, fi2.h_in_b);
        let mut cdf_natural = CDFContext::new(fi2.base_q_idx);
        let natural_bytes = {
            let mut iter = ti.tile_iter_mut(&mut fs2, &mut blocks2);
            let mut ctx = iter.next().unwrap();
            drop(iter);
            let (mut writer, _stats): (WriterBase<NativeEnc>, _) =
                encode_tile(&fi2, &mut ctx.ts, &mut cdf_natural, &mut ctx.tb, &inter_cfg2);
            writer.done()
        };

        // Pass 2: hybrid replay with EMPTY plan.
        let stego_bytes_empty = finalize_replay(&recorder, &OverrideMap::new());

        // Round-trip byte-identity.
        assert_eq!(
            stego_bytes_empty, natural_bytes,
            "empty-plan hybrid replay must match natural rav1e encode byte-for-byte"
        );
    }

    /// W3.8.6 Test B — override fires on the wire: non-empty plan
    /// produces different bytes than natural, but with the SAME LENGTH.
    /// Length invariance is load-bearing for the OBU-splice strategy
    /// (per rav1e-hook-sites.md § 3.1).
    #[test]
    fn w386_round_trip_override_changes_bytes_same_length() {
        use phasm_rav1e::context::CDFContext;
        use phasm_rav1e::phasm_stego::FrameBlocks;

        // Pass 1: capture recorder.
        let (fi, mut fs, inter_cfg) = setup_rav1e_state();
        let mut blocks = FrameBlocks::new(fi.w_in_b, fi.h_in_b);
        let mut cdf = CDFContext::new(fi.base_q_idx);
        let ti = &fi.sequence.tiling;
        let recorder = {
            let mut iter = ti.tile_iter_mut(&mut fs, &mut blocks);
            let mut ctx = iter.next().unwrap();
            drop(iter);
            let (recorder, _stats): (WriterBase<WriterRecorder>, _) =
                encode_tile(&fi, &mut ctx.ts, &mut cdf, &mut ctx.tb, &inter_cfg);
            recorder
        };

        let n_l1 = recorder.phasm_bit_positions().len();
        assert!(n_l1 > 0, "must have at least one L(1) emission to test overrides");

        // Replay with empty plan → "natural" bytes.
        let natural_bytes = finalize_replay(&recorder, &OverrideMap::new());

        // Replay with override at cursor 0 (flip first L(1) emission).
        let first_natural_val = recorder.phasm_bit_positions()[0].1;
        let flipped_val = 1 - first_natural_val;
        let mut plan = OverrideMap::new();
        plan.set(0, flipped_val);
        let stego_bytes = finalize_replay(&recorder, &plan);

        // Override produces DIFFERENT bytes.
        assert_ne!(
            stego_bytes, natural_bytes,
            "override at cursor 0 must produce different bytes than empty plan"
        );

        // But SAME LENGTH (load-bearing invariant for OBU splice).
        assert_eq!(
            stego_bytes.len(), natural_bytes.len(),
            "Tier 1 50/50 override must preserve tile_group byte length \
             (range coder consumes same bit count for either value at 50/50 probability)"
        );
    }

    /// W3.8.6 Test C — multiple-override capacity. Flip every 3rd L(1)
    /// emission; verify bytes differ AND length is preserved across
    /// many override sites. Stress test for the length-invariance
    /// claim.
    #[test]
    fn w386_round_trip_multiple_overrides_preserve_length() {
        use phasm_rav1e::context::CDFContext;
        use phasm_rav1e::phasm_stego::FrameBlocks;

        let (fi, mut fs, inter_cfg) = setup_rav1e_state();
        let mut blocks = FrameBlocks::new(fi.w_in_b, fi.h_in_b);
        let mut cdf = CDFContext::new(fi.base_q_idx);
        let ti = &fi.sequence.tiling;
        let recorder = {
            let mut iter = ti.tile_iter_mut(&mut fs, &mut blocks);
            let mut ctx = iter.next().unwrap();
            drop(iter);
            let (recorder, _stats): (WriterBase<WriterRecorder>, _) =
                encode_tile(&fi, &mut ctx.ts, &mut cdf, &mut ctx.tb, &inter_cfg);
            recorder
        };

        let bit_positions = recorder.phasm_bit_positions().to_vec();
        let n_l1 = bit_positions.len();
        assert!(n_l1 >= 9, "test needs ≥9 L(1) emissions; got {}", n_l1);

        let natural_bytes = finalize_replay(&recorder, &OverrideMap::new());

        // Flip every 3rd emission's natural value.
        let mut plan = OverrideMap::new();
        for (cursor, &(_, natural)) in bit_positions.iter().enumerate().step_by(3) {
            plan.set(cursor as u64, 1 - natural);
        }
        let stego_bytes = finalize_replay(&recorder, &plan);

        assert_ne!(
            stego_bytes, natural_bytes,
            "multi-position overrides must produce different bytes"
        );
        assert_eq!(
            stego_bytes.len(),
            natural_bytes.len(),
            "length invariance must hold across many override sites"
        );
    }
}

