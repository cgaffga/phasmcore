// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Encoder integration hook for encode-time CABAC stego (Phase 6D.8).
//
// The `StegoMbHook` trait gives the per-MB encoder code an
// optional pre-entropy-coding callback at every residual block +
// MVD emit point. When a hook is registered:
//   - Pass 1 (capacity): hook is a position-counter/logger; it
//     enumerates positions but does NOT mutate the buffer. Encoder
//     output bytes are typically discarded after Pass 1.
//   - Pass 3 (inject): hook calls `apply_*_overrides` to flip
//     coefficient signs / suffix LSBs / MVD values per the STC
//     plan. Encoder output bytes are the final stego bitstream.
//
// **Critical invariant**: with no hook (or with a no-op hook) the
// encoder's output must be byte-identical to today's pre-6D.8
// behavior. The hook is opt-in only.
//
// **Where the hook fires**: between quantization (or MVD
// derivation) and the entropy emit. Hook may mutate the
// scan_coeffs buffer in place; both reconstruction and entropy
// then see the modified values. Same for MVD.

use super::inject::{
    apply_coeff_sign_overrides, apply_coeff_suffix_lsb_overrides, MvdSlot,
};
use super::orchestrate::ResidualPathKind;
use super::{BinKind, BitInjector};

/// Encoder-side stego hook. Implementations can either count
/// positions (Pass 1) or apply overrides (Pass 3).
///
/// **Concurrency**: invoked single-threaded per GOP. Within a GOP
/// the encoder visits MBs in raster order; the hook sees them in
/// the same order. Across GOPs (multi-GOP encode), each GOP's
/// hook lives on its own rayon worker — hence the `Send` bound.
///
/// **Debug bound**: required so the `Encoder` struct (which holds
/// an optional `Box<dyn StegoMbHook>`) can keep its
/// `#[derive(Debug)]`. Implementations that don't have anything
/// useful to print can hand-impl with `finish_non_exhaustive()`.
pub trait StegoMbHook: Send + std::fmt::Debug {
    /// Called by the encoder after quantization, before entropy
    /// coding for one residual block. Hook may mutate
    /// `scan_coeffs` in place.
    fn on_residual_block(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        scan_coeffs: &mut [i32],
        start_idx: usize,
        end_idx: usize,
        path_kind: ResidualPathKind,
    );

    /// Called by the encoder after MVD derivation, before MVD
    /// entropy coding. Hook may modify `slot.value` in place.
    fn on_mvd_slot(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        slot: &mut MvdSlot,
    );

    /// Optional: drain accumulated cover state from a Pass 1
    /// `PositionLoggerHook`. Default returns `None` (non-loggers
    /// have nothing to drain). Used by the orchestration driver
    /// to recover the `GopCover` from a `Box<dyn StegoMbHook>`
    /// after Pass 1 completes — avoids unsafe downcasting.
    fn take_cover_if_logger(&mut self) -> Option<super::orchestrate::GopCover> {
        None
    }

    /// Phase 6F.2(j) — drain per-MVD-position metadata captured
    /// alongside the cover. Default returns an empty Vec for
    /// non-loggers. Aligned by index with
    /// `take_cover_if_logger().cover.mvd_sign_bypass.positions`.
    fn take_mvd_meta_if_logger(&mut self) -> Vec<MvdPositionMeta> {
        Vec::new()
    }

    /// Phase 6F.2 — begin a per-MB MVD-position savepoint.
    /// Loggers should remember their current MVD-position offsets
    /// so a subsequent `rollback_mvd_for_mb` can retract any
    /// `on_mvd_slot` pushes. Default no-op for non-logger hooks.
    /// Encoder calls this at the start of each P-MB / B-MB before
    /// `apply_mvd_hook_to_choice` fires.
    fn begin_mvd_for_mb(&mut self) {}

    /// Phase 6F.2 — commit any pending per-MB MVD positions into
    /// the persistent cover. Encoder calls this after the MB has
    /// definitively committed to emitting MVDs in the bitstream
    /// (i.e. NOT P_SKIP, NOT intra-in-P). For PositionLoggerHook
    /// this just clears the savepoint; for plan-based injection
    /// hooks it's a no-op.
    fn commit_mvd_for_mb(&mut self) {}

    /// Phase 6F.2 — discard any pending per-MB MVD positions
    /// pushed since the last `begin_mvd_for_mb`. Encoder calls
    /// this when the MB ended up as P_SKIP or intra-in-P — those
    /// emit no MVDs in the bitstream, so any positions logged by
    /// the MVD hook were phantoms.
    fn rollback_mvd_for_mb(&mut self) {}

    /// Phase 6F.2(k).2 — query the planned MVD-sign override at
    /// CABAC emit time. Returns `Some(b)` if the position has a
    /// stego plan flip; the encoder writes `b` to the bypass sign
    /// bin instead of `slot.value < 0`. Returns `None` for
    /// unplanned positions (encoder writes the natural sign).
    ///
    /// Crucially, this DOES NOT mutate `slot.value`. The encoder's
    /// `mv_grid` + motion compensation + neighbor predictors all
    /// see the original (pre-injection) MV. Only the bypass sign
    /// bin written to the emitted bitstream differs from a clean
    /// encode. By construction this avoids the predictor cascade
    /// that broke the §6F.2(j) cascade-modeling approach.
    ///
    /// Default returns `None` (non-injection hooks): the encoder
    /// emits natural MVD sign bins.
    fn mvd_sign_override(
        &mut self,
        _frame_idx: u32,
        _mb_addr: u32,
        _slot: &MvdSlot,
    ) -> Option<u8> {
        None
    }
}

/// Pass 3 implementation: forwards every emit-site call to the
/// matching `apply_*_overrides` primitive, using a [`BitInjector`]
/// backed by the precomputed `DomainPlan`.
///
/// Wraps a single `BitInjector` reference; the injector is
/// typically a `PlanInjector` built from the Pass 2 STC plan.
pub struct InjectionHook<I: BitInjector> {
    injector: I,
}

impl<I: BitInjector> InjectionHook<I> {
    pub fn new(injector: I) -> Self {
        Self { injector }
    }

    /// Consume the hook and return the underlying injector.
    pub fn into_injector(self) -> I {
        self.injector
    }
}

impl<I: BitInjector> std::fmt::Debug for InjectionHook<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InjectionHook").finish_non_exhaustive()
    }
}

impl<I: BitInjector + Send> StegoMbHook for InjectionHook<I> {
    fn on_residual_block(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        scan_coeffs: &mut [i32],
        start_idx: usize,
        end_idx: usize,
        path_kind: ResidualPathKind,
    ) {
        // Sign domain: apply sign overrides. Magnitudes preserved.
        apply_coeff_sign_overrides(
            scan_coeffs, start_idx, end_idx, frame_idx, mb_addr,
            |ci| path_kind.path(ci, BinKind::Sign),
            &mut self.injector,
        );
        // Suffix-LSB domain: apply ±1 magnitude flips at eligible
        // positions (|coeff| ≥ 16). Threshold-aware direction.
        apply_coeff_suffix_lsb_overrides(
            scan_coeffs, start_idx, end_idx, frame_idx, mb_addr,
            |ci| path_kind.path(ci, BinKind::SuffixLsb),
            &mut self.injector,
        );
    }

    fn on_mvd_slot(
        &mut self,
        _frame_idx: u32,
        _mb_addr: u32,
        _slot: &mut MvdSlot,
    ) {
        // Phase 6F.2(k).2 — InjectionHook NO LONGER mutates
        // slot.value at the apply_mvd_hook_to_choice site. The
        // sign-bit override is now applied at CABAC bypass-emit
        // time via `mvd_sign_override` (queried by the encoder
        // immediately before writing the bypass sign bin).
        //
        // This decouples "encoder-internal MVD value" (which feeds
        // mv_grid + MC + neighbor predictors) from "the sign bit
        // written to the bitstream". The mv_grid stays at the
        // encoder's natural value → no cascade.
        //
        // Suffix-LSB MVD injection is also disabled here for the
        // same reason: a magnitude-LSB flip changes |MVD| by 1
        // which propagates through the median predictor. v1.0
        // bitstream-mod path uses sign-only (the dominant
        // bypass-bin family for stealth-fingerprint purposes).
    }

    fn mvd_sign_override(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        slot: &MvdSlot,
    ) -> Option<u8> {
        // Compute the PositionKey the planner used for this slot's
        // sign bin and consult the injector. `BitInjector::override_bit`
        // returns `Some(plan_bit)` for planned positions.
        if slot.value == 0 {
            // Spec: zero MVDs emit no sign bin. Override is moot.
            return None;
        }
        use super::hook::{EmbedDomain, PositionKey, SyntaxPath, BinKind};
        let path = SyntaxPath::Mvd {
            list: slot.list,
            partition: slot.partition,
            axis: slot.axis,
            kind: BinKind::Sign,
        };
        let key = PositionKey::new(frame_idx, mb_addr, EmbedDomain::MvdSignBypass, path);
        self.injector.override_bit(key)
    }
}

/// Pass 1 implementation: enumerates positions per domain and
/// pushes them into a [`super::orchestrate::GopCover`]. Does NOT
/// mutate `scan_coeffs` or `slot`.
///
/// Builds the cover incrementally as the encoder visits each MB
/// in raster order. After encoding the entire GOP, the caller
/// extracts the cover via [`PositionLoggerHook::take_cover`] and
/// passes it to Pass 1.5 (`split_message_per_domain`).
/// Phase 6F.2(j) — per-MVD-position metadata captured alongside
/// the cover. Aligned by INDEX with `cover.cover.mvd_sign_bypass`
/// (one entry per logged sign position; empty when the MVD-sign
/// position list is empty).
///
/// Used by `cascade_safety::analyze_safe_mvd_subset` to build the
/// dependency-graph + run criterion-C greedy. The walker side
/// captures the same shape from `decode_mvd_with_bin0_inc` output
/// so encoder + decoder run the safe-set computation on identical
/// inputs.
#[derive(Copy, Clone, Debug, Default)]
pub struct MvdPositionMeta {
    /// Absolute MVD value at hook-fire time. Encoder side: pre-
    /// or post-injection depending on hook order (PositionLogger
    /// fires post-injection in InjectAndLogHook). Walker side:
    /// the value decoded from the bitstream.
    pub magnitude: u32,
    /// Frame-relative MB address (mb_y * mb_w + mb_x).
    pub mb_addr: u32,
    /// Frame index in encoder/walker bitstream order
    /// (same semantics as `Encoder::stego_frame_idx`).
    pub frame_idx: u32,
    /// Partition value packed into `MvdSlot::partition` (0 for
    /// P_L0_16x16; for sub-MBs `sub_mb_idx * 4 + sub_part_idx`).
    pub partition: u8,
    /// 0 = X, 1 = Y. Matches the layout of `Axis`.
    pub axis: u8,
}

#[derive(Debug)]
pub struct PositionLoggerHook {
    cover: super::orchestrate::GopCover,
    /// Phase 6F.2 — per-MB savepoint for the MVD position lengths.
    /// `Some((sign_len, suffix_len))` between `begin_mvd_for_mb`
    /// and the matching `commit_mvd_for_mb` / `rollback_mvd_for_mb`;
    /// `None` outside an MB-MVD transaction. Encoder must always
    /// pair begin with exactly one commit/rollback per MB or
    /// positions will be miscategorised.
    mvd_savepoint: Option<(usize, usize)>,
    /// Phase 6F.2(j) — per-position metadata aligned with
    /// `cover.cover.mvd_sign_bypass`. Drained alongside the cover
    /// via `take_mvd_meta()`. Begin/commit/rollback semantics
    /// mirror `mvd_savepoint`: rollback truncates this vector to
    /// the savepoint's `sign_len`.
    mvd_meta: Vec<MvdPositionMeta>,
    /// Savepoint for `mvd_meta.len()` taken at `begin_mvd_for_mb`.
    /// Distinct from `mvd_savepoint`'s `sign_len` only because the
    /// `mvd_meta` vector is per-sign-position (no parallel suffix
    /// vector). Always equal to `mvd_savepoint.0` in practice
    /// since both update on the same `on_mvd_slot` call.
    mvd_meta_savepoint: Option<usize>,
}

impl Default for PositionLoggerHook {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionLoggerHook {
    pub fn new() -> Self {
        Self {
            cover: super::orchestrate::GopCover::default(),
            mvd_savepoint: None,
            mvd_meta: Vec::new(),
            mvd_meta_savepoint: None,
        }
    }

    pub fn take_cover(&mut self) -> super::orchestrate::GopCover {
        std::mem::take(&mut self.cover)
    }

    /// Phase 6F.2(j) — drain the per-position MVD metadata captured
    /// alongside the cover. Returned vector is index-aligned with
    /// `take_cover().cover.mvd_sign_bypass.positions`.
    pub fn take_mvd_meta(&mut self) -> Vec<MvdPositionMeta> {
        std::mem::take(&mut self.mvd_meta)
    }
}

/// Phase 6D.8 §30D-C composite hook for Pass 1B of the 3-pass
/// orchestrator. Routes:
/// - `on_mvd_slot` → `InjectionHook` (applies Pass-2A's MVD plan,
///   modifying slot.value so encoder's MC + recon use FINAL MV).
/// - `on_residual_block` → `PositionLoggerHook` (records the
///   POST-MVD-injection residual cover for Stage B planning).
///
/// `take_cover_if_logger` returns the residual logger's cover only
/// (MVD positions stay empty in the returned GopCover; Pass 2A
/// already used the Pass 1 MVD cover for its plan).
pub struct InjectAndLogHook<I: BitInjector> {
    inject: InjectionHook<I>,
    logger: PositionLoggerHook,
}

impl<I: BitInjector> InjectAndLogHook<I> {
    pub fn new(injector: I) -> Self {
        Self {
            inject: InjectionHook::new(injector),
            logger: PositionLoggerHook::new(),
        }
    }
}

impl<I: BitInjector> std::fmt::Debug for InjectAndLogHook<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InjectAndLogHook").finish_non_exhaustive()
    }
}

impl<I: BitInjector + Send> StegoMbHook for InjectAndLogHook<I> {
    fn on_residual_block(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        scan_coeffs: &mut [i32],
        start_idx: usize,
        end_idx: usize,
        path_kind: ResidualPathKind,
    ) {
        // Log only — no residual injection in Pass 1B (residual
        // plan doesn't exist yet, computed in Pass 2B from this
        // logger's cover).
        self.logger.on_residual_block(
            frame_idx, mb_addr, scan_coeffs, start_idx, end_idx, path_kind,
        );
    }

    fn on_mvd_slot(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        slot: &mut super::inject::MvdSlot,
    ) {
        // Phase 6F.2(k).2 — InjectionHook no longer mutates
        // slot.value here (sign override applied at CABAC emit
        // time). InjectAndLogHook just logs the slot for cover
        // construction. mv_grid stays at original → no cascade.
        self.inject.on_mvd_slot(frame_idx, mb_addr, slot);
        self.logger.on_mvd_slot(frame_idx, mb_addr, slot);
    }

    fn mvd_sign_override(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        slot: &super::inject::MvdSlot,
    ) -> Option<u8> {
        // Forward to the inner InjectionHook so the planned sign
        // override is applied at CABAC emit time.
        self.inject.mvd_sign_override(frame_idx, mb_addr, slot)
    }

    fn begin_mvd_for_mb(&mut self) {
        self.logger.begin_mvd_for_mb();
    }

    fn commit_mvd_for_mb(&mut self) {
        self.logger.commit_mvd_for_mb();
    }

    fn rollback_mvd_for_mb(&mut self) {
        self.logger.rollback_mvd_for_mb();
    }

    fn take_cover_if_logger(&mut self) -> Option<super::orchestrate::GopCover> {
        Some(self.logger.take_cover())
    }

    fn take_mvd_meta_if_logger(&mut self) -> Vec<MvdPositionMeta> {
        self.logger.take_mvd_meta()
    }
}

impl StegoMbHook for PositionLoggerHook {
    fn take_cover_if_logger(&mut self) -> Option<super::orchestrate::GopCover> {
        Some(self.take_cover())
    }

    fn take_mvd_meta_if_logger(&mut self) -> Vec<MvdPositionMeta> {
        self.take_mvd_meta()
    }

    fn begin_mvd_for_mb(&mut self) {
        self.mvd_savepoint = Some((
            self.cover.cover.mvd_sign_bypass.len(),
            self.cover.cover.mvd_suffix_lsb.len(),
        ));
        self.mvd_meta_savepoint = Some(self.mvd_meta.len());
    }

    fn commit_mvd_for_mb(&mut self) {
        // Drop the savepoint; positions stay in the cover.
        self.mvd_savepoint = None;
        self.mvd_meta_savepoint = None;
    }

    fn rollback_mvd_for_mb(&mut self) {
        if let Some((sign_len, suffix_len)) = self.mvd_savepoint.take() {
            self.cover.cover.mvd_sign_bypass.truncate(sign_len);
            self.cover.cover.mvd_suffix_lsb.truncate(suffix_len);
            self.cover.costs.mvd_sign_bypass.truncate(sign_len);
            self.cover.costs.mvd_suffix_lsb.truncate(suffix_len);
        }
        if let Some(meta_len) = self.mvd_meta_savepoint.take() {
            self.mvd_meta.truncate(meta_len);
        }
    }

    fn on_residual_block(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        scan_coeffs: &mut [i32],
        start_idx: usize,
        end_idx: usize,
        path_kind: ResidualPathKind,
    ) {
        use super::cost_model::{coeff_sign_cost_vec, coeff_suffix_lsb_cost_vec, PositionCostCtx};
        use super::{
            enumerate_coeff_sign_positions, enumerate_coeff_suffix_lsb_positions,
            extract_coeff_sign_bits, extract_coeff_suffix_lsb_bits,
        };
        let ctx = PositionCostCtx::new(frame_idx, mb_addr);

        // CoeffSignBypass.
        let positions = enumerate_coeff_sign_positions(
            scan_coeffs, start_idx, end_idx, frame_idx, mb_addr,
            |ci| path_kind.path(ci, BinKind::Sign),
        );
        let bits = extract_coeff_sign_bits(scan_coeffs, start_idx, end_idx);
        let costs = coeff_sign_cost_vec(scan_coeffs, start_idx, end_idx, &ctx);
        for ((p, b), c) in positions.iter().zip(bits.iter()).zip(costs.iter()) {
            self.cover.cover.coeff_sign_bypass.push(*b, *p);
            self.cover.costs.coeff_sign_bypass.push(*c);
        }

        // CoeffSuffixLsb.
        let positions = enumerate_coeff_suffix_lsb_positions(
            scan_coeffs, start_idx, end_idx, frame_idx, mb_addr,
            |ci| path_kind.path(ci, BinKind::SuffixLsb),
        );
        let bits = extract_coeff_suffix_lsb_bits(scan_coeffs, start_idx, end_idx);
        let costs = coeff_suffix_lsb_cost_vec(scan_coeffs, start_idx, end_idx, &ctx);
        for ((p, b), c) in positions.iter().zip(bits.iter()).zip(costs.iter()) {
            self.cover.cover.coeff_suffix_lsb.push(*b, *p);
            self.cover.costs.coeff_suffix_lsb.push(*c);
        }
    }

    fn on_mvd_slot(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        slot: &mut MvdSlot,
    ) {
        use super::cost_model::{mvd_sign_cost_vec, mvd_suffix_lsb_cost_vec, PositionCostCtx};
        use super::{
            enumerate_mvd_sign_positions, enumerate_mvd_suffix_lsb_positions,
            extract_mvd_sign_bits, extract_mvd_suffix_lsb_bits,
        };
        let single = [*slot];
        let ctx = PositionCostCtx::new(frame_idx, mb_addr);

        // MvdSignBypass.
        let positions = enumerate_mvd_sign_positions(&single, frame_idx, mb_addr);
        let bits = extract_mvd_sign_bits(&single);
        let costs = mvd_sign_cost_vec(&single, &ctx);
        let pre_sign_len = self.cover.cover.mvd_sign_bypass.len();
        for ((p, b), c) in positions.iter().zip(bits.iter()).zip(costs.iter()) {
            self.cover.cover.mvd_sign_bypass.push(*b, *p);
            self.cover.costs.mvd_sign_bypass.push(*c);
        }
        // Phase 6F.2(j) — capture per-position metadata aligned with
        // the mvd_sign_bypass entry that was just pushed (if any).
        // `enumerate_mvd_sign_positions` filters zero-valued slots,
        // so positions.len() ∈ {0, 1} for a single-slot input.
        let pushed = self.cover.cover.mvd_sign_bypass.len() - pre_sign_len;
        if pushed > 0 {
            use super::Axis;
            self.mvd_meta.push(MvdPositionMeta {
                magnitude: slot.value.unsigned_abs(),
                mb_addr,
                frame_idx,
                partition: slot.partition,
                axis: match slot.axis { Axis::X => 0, Axis::Y => 1 },
            });
        }

        // MvdSuffixLsb.
        let positions = enumerate_mvd_suffix_lsb_positions(&single, frame_idx, mb_addr);
        let bits = extract_mvd_suffix_lsb_bits(&single);
        let costs = mvd_suffix_lsb_cost_vec(&single, &ctx);
        for ((p, b), c) in positions.iter().zip(bits.iter()).zip(costs.iter()) {
            self.cover.cover.mvd_suffix_lsb.push(*b, *p);
            self.cover.costs.mvd_suffix_lsb.push(*c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::Axis;

    #[test]
    fn position_logger_residual_block_collects_per_domain() {
        let mut hook = PositionLoggerHook::new();
        let mut scan = vec![0i32; 16];
        scan[0] = 5; scan[3] = -7; scan[6] = 20; // last is suffix-eligible
        hook.on_residual_block(
            0, 0, &mut scan, 0, 15,
            ResidualPathKind::Luma4x4 { block_idx: 0 },
        );
        let cover = hook.take_cover();
        // 3 nonzero coeffs → 3 sign positions
        assert_eq!(cover.cover.coeff_sign_bypass.len(), 3);
        // 1 |coeff| ≥ 16 → 1 suffix LSB position
        assert_eq!(cover.cover.coeff_suffix_lsb.len(), 1);
    }

    #[test]
    fn position_logger_mvd_slot_collects_per_domain() {
        let mut hook = PositionLoggerHook::new();
        let mut slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 5 };
        hook.on_mvd_slot(0, 0, &mut slot);
        let cover = hook.take_cover();
        // |mvd|=5 → 1 sign position, no suffix (|mvd|<9)
        assert_eq!(cover.cover.mvd_sign_bypass.len(), 1);
        assert_eq!(cover.cover.mvd_suffix_lsb.len(), 0);

        let mut hook = PositionLoggerHook::new();
        let mut slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 15 };
        hook.on_mvd_slot(0, 0, &mut slot);
        let cover = hook.take_cover();
        // |mvd|=15 → 1 sign + 1 suffix
        assert_eq!(cover.cover.mvd_sign_bypass.len(), 1);
        assert_eq!(cover.cover.mvd_suffix_lsb.len(), 1);
    }

    #[test]
    fn position_logger_zero_inputs_emit_no_positions() {
        let mut hook = PositionLoggerHook::new();
        let mut scan = vec![0i32; 16];
        hook.on_residual_block(
            0, 0, &mut scan, 0, 15,
            ResidualPathKind::Luma4x4 { block_idx: 0 },
        );
        let mut slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 0 };
        hook.on_mvd_slot(0, 0, &mut slot);
        let cover = hook.take_cover();
        assert_eq!(cover.cover.coeff_sign_bypass.len(), 0);
        assert_eq!(cover.cover.coeff_suffix_lsb.len(), 0);
        assert_eq!(cover.cover.mvd_sign_bypass.len(), 0);
        assert_eq!(cover.cover.mvd_suffix_lsb.len(), 0);
    }

    /// Force-flip injector that always returns `Some(target_bit)`.
    struct ForceBit(u8);
    impl BitInjector for ForceBit {
        fn override_bit(&mut self, _key: super::super::PositionKey) -> Option<u8> {
            Some(self.0)
        }
    }

    #[test]
    fn injection_hook_residual_block_flips_signs() {
        let mut hook = InjectionHook::new(ForceBit(1)); // force negative
        let mut scan = vec![0i32; 16];
        scan[0] = 5; scan[3] = 7; scan[6] = -2;
        hook.on_residual_block(
            0, 0, &mut scan, 0, 15,
            ResidualPathKind::Luma4x4 { block_idx: 0 },
        );
        // Force=1 ⇒ all become negative.
        assert_eq!(scan[0], -5);
        assert_eq!(scan[3], -7);
        assert_eq!(scan[6], -2);
    }

    #[test]
    fn injection_hook_residual_block_flips_suffix_lsb() {
        let mut hook = InjectionHook::new(ForceBit(0));
        let mut scan = vec![0i32; 16];
        // |coeff|=20 → cover suffix LSB = NOT(20&1) = 1. Target=0 ⇒ flip.
        scan[0] = 20;
        hook.on_residual_block(
            0, 0, &mut scan, 0, 15,
            ResidualPathKind::Luma4x4 { block_idx: 0 },
        );
        // Sign forced to 0 (positive). Suffix LSB forced to 0 ⇒ |20|→|19|.
        assert!(scan[0] > 0, "sign flipped to positive");
        assert_eq!(scan[0].unsigned_abs(), 19, "suffix LSB flip ±1");
    }

    /// Phase 6F.2(k).2 — InjectionHook NO LONGER mutates slot.value
    /// at the apply_mvd_hook_to_choice site. The sign-bit override
    /// is applied at CABAC emit time via `mvd_sign_override`. This
    /// test was previously asserting the old in-place-mutate
    /// behavior; updated to confirm the new no-op semantics on
    /// `on_mvd_slot` AND the override-bit query semantics.
    #[test]
    fn injection_hook_mvd_slot_does_not_mutate() {
        let mut hook = InjectionHook::new(ForceBit(0));
        let mut slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: -5 };
        hook.on_mvd_slot(0, 0, &mut slot);
        assert_eq!(slot.value, -5,
            "InjectionHook::on_mvd_slot must NOT mutate slot.value (decouples encoder mv_grid from bitstream sign bit)");
    }

    #[test]
    fn injection_hook_mvd_sign_override_returns_planned_bit() {
        let mut hook = InjectionHook::new(ForceBit(0));
        let slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: -5 };
        let r = hook.mvd_sign_override(0, 0, &slot);
        assert_eq!(r, Some(0),
            "mvd_sign_override must return the planned bit value");
    }

    #[test]
    fn injection_hook_mvd_sign_override_zero_returns_none() {
        let mut hook = InjectionHook::new(ForceBit(1));
        let slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 0 };
        let r = hook.mvd_sign_override(0, 0, &slot);
        assert_eq!(r, None,
            "mvd=0 has no sign bypass bin in spec; override is a no-op");
    }

    #[test]
    fn position_logger_does_not_mutate_inputs() {
        let mut hook = PositionLoggerHook::new();
        let mut scan = vec![0i32; 16];
        scan[0] = 5; scan[3] = -7;
        let original = scan.clone();
        hook.on_residual_block(
            0, 0, &mut scan, 0, 15,
            ResidualPathKind::Luma4x4 { block_idx: 0 },
        );
        assert_eq!(scan, original, "position logger must not mutate inputs");
    }
}
