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

    /// §long-form-stego Phase 2 — drain a `PositionCountingHook`'s
    /// running totals as `[coeff_sign, coeff_suffix, mvd_sign,
    /// mvd_suffix]`. Default returns `None` for non-counter hooks.
    /// Used by `pass1_count_per_gop_4domain` to harvest per-GOP
    /// counts after fresh-per-GOP hook install.
    fn take_counts_if_counter(&mut self) -> Option<[usize; 4]> {
        None
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

    /// #540.1 — query the planned MvdSuffixLsb magnitude override at
    /// CABAC emit time. Returns `Some(target_abs)` when the position
    /// has a stego plan flip; the encoder emits a UEG3 prefix +
    /// suffix for `target_abs` (NOT `|slot.value|`). Returns `None`
    /// for unplanned positions.
    ///
    /// Like [`Self::mvd_sign_override`], this DOES NOT mutate
    /// `slot.value`. The encoder's `mv_grid` + motion compensation +
    /// neighbor predictors all see the natural pre-injection MVD →
    /// no cascade. The walker reading the wire decodes `target_abs`
    /// (with `slot.value`'s natural sign) → MV differs by ±1 on the
    /// flipped axis. This is the wire-only counterpart of the
    /// `apply_mvd_suffix_lsb_overrides` / `on_mvd_slot` mutation
    /// path; both achieve the same wire output but the wire-only
    /// path avoids the encoder-state cascade that the mutation path
    /// would otherwise create (which forced the cascade-safety
    /// criterion-C filter).
    ///
    /// Caller (encoder MVD emit site) MUST:
    /// 1. Pass `Some(target_abs)` as `abs_override` to
    ///    `encode_mvd_with_bin0_inc_overrides`.
    /// 2. Use `target_abs` (NOT `|slot.value|`) when calling
    ///    `current_mvd.fill_region` so within-MB downstream bin0
    ///    `ctxIdxInc` stays symmetric with the walker.
    ///
    /// Default returns `None` (non-injection hooks): the encoder
    /// emits natural MVD magnitudes.
    fn mvd_suffix_lsb_abs_override(
        &mut self,
        _frame_idx: u32,
        _mb_addr: u32,
        _slot: &MvdSlot,
    ) -> Option<u32> {
        None
    }
}

/// Pass 3 implementation: forwards every emit-site call to the
/// matching `apply_*_overrides` primitive, using a [`BitInjector`]
/// backed by the precomputed `DomainPlan`.
///
/// Wraps a single `BitInjector` reference; the injector is
/// typically a `PlanInjector` built from the Pass 2 STC plan.
///
/// **§6E-A5(d).4** — `mvd_msl_safe_gate` is the gate-set for
/// magnitude-LSB MVD overrides. When `None` (default), `on_mvd_slot`
/// is a no-op regardless of what the injector returns — matches the
/// pre-d.4 sign-only-bitstream-mod production behavior. When
/// `Some(set)`, the hook only fires the magnitude flip when the
/// MVD's MvdSuffixLsb PositionKey is in `set`. The shadow encoder
/// populates this set from `shadow_states` (positions
/// upstream-filtered by `cascade_safety::analyze_safe_mvd_subset`).
pub struct InjectionHook<I: BitInjector> {
    injector: I,
    mvd_msl_safe_gate: Option<std::collections::HashSet<super::hook::PositionKey>>,
}

impl<I: BitInjector> InjectionHook<I> {
    pub fn new(injector: I) -> Self {
        Self {
            injector,
            mvd_msl_safe_gate: None,
        }
    }

    /// §6E-A5(d).4 — install the cascade-safe MvdSuffixLsb gate-set.
    /// `on_mvd_slot` will fire magnitude-LSB flips only at
    /// PositionKeys present in `keys`. Caller computes the set from
    /// shadow's `priority_slots_all4_safe(... safe_msl=Some(...))`
    /// output, restricted to the MvdSuffixLsb-domain slots.
    pub fn set_mvd_msl_safe_gate(
        &mut self,
        keys: std::collections::HashSet<super::hook::PositionKey>,
    ) {
        self.mvd_msl_safe_gate = Some(keys);
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
        // #540.2 — RETIRED. Pre-port behavior:
        //
        //   if gate_set: mutate slot.value by ±1 to align magnitude
        //   LSB with planned bit. This propagated through `mv_grid` +
        //   MC + neighbor predictor (cascade), requiring the
        //   cascade-safety predicate to bound the propagation.
        //
        // Post-port behavior (wire-only): MvdSuffixLsb flips happen
        // exclusively at CABAC emit via [`Self::mvd_suffix_lsb_abs_override`]
        // (analogous to the existing `mvd_sign_override` path).
        // `slot.value` is now NEVER mutated by InjectionHook, so
        // `mv_grid` always sees the natural pre-injection MV — no
        // cascade. The walker reads the override magnitude from the
        // wire and reconstructs its own (different) MV; this is the
        // visual quality cost of MvdSuffixLsb stego, identical to
        // OH264 post-#539 (cascade-break deletion).
        //
        // No-op kept to satisfy trait + preserve existing call sites
        // in `invoke_stego_mvd_hook`.
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

    /// #540.1 + #540.2 — wire-only MvdSuffixLsb path. Mirror of
    /// `mvd_sign_override` for the magnitude domain.
    ///
    /// Returns the target magnitude (NOT a bit) the wire should emit
    /// for this slot. Walker convention: plan bit is computed against
    /// `suffix_lsb_bit_for_magnitude(abs) = (abs & 1) ^ 1`.
    ///
    /// Fires for ALL planned positions (primary STC + shadow). The
    /// retired [`Self::on_mvd_slot`] mutation path is no longer
    /// active — both paths converge on emit-time abs override, which
    /// keeps `slot.value` + `mv_grid` clean (= no cascade).
    fn mvd_suffix_lsb_abs_override(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        slot: &MvdSlot,
    ) -> Option<u32> {
        if slot.value == 0 {
            return None;
        }
        let abs = slot.value.unsigned_abs();
        // MvdSuffixLsb is only emitted on the wire when |MVD| >= 9.
        // Below this threshold, there's no suffix-LSB position to
        // override and the walker doesn't enumerate one.
        if abs < super::inject::MVD_SUFFIX_LSB_THRESHOLD {
            return None;
        }
        use super::hook::{EmbedDomain, PositionKey, SyntaxPath, BinKind};
        let path = SyntaxPath::Mvd {
            list: slot.list,
            partition: slot.partition,
            axis: slot.axis,
            kind: BinKind::SuffixLsb,
        };
        let key = PositionKey::new(frame_idx, mb_addr, EmbedDomain::MvdSuffixLsb, path);
        let plan_bit = self.injector.override_bit(key)?;
        // Walker convention: cover_bit = (abs & 1) ^ 1.
        let cover_bit = ((abs & 1) ^ 1) as u8;
        if plan_bit == cover_bit {
            return None;
        }
        // Direction: prefer abs-1 (smaller perturbation), except at
        // the eligibility boundary (abs == threshold = 9) where
        // we must go +1 to keep target_abs >= 9. Mirror of
        // `flipped_magnitude` in `apply_mvd_suffix_lsb_overrides`.
        let new_abs = super::inject::mvd_flipped_magnitude(abs);
        Some(new_abs)
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

/// §long-form-stego Phase 2 — counting-only Pass 1 hook.
///
/// Mirrors `PositionLoggerHook`'s position-enumeration logic but
/// only RUNS the enumerator and counts the returned length per
/// domain, without storing bits / costs / PositionKey values. Used
/// by `pass1_count_per_gop_4domain` to map seg_idx → GOP range
/// without an O(n) materialization.
///
/// The four counters are running totals across the entire encode.
/// The orchestrator driver snapshots them at each GOP boundary;
/// per-GOP counts are the diff between consecutive snapshots.
///
/// Memory: O(1) — four `usize` counters plus the `mvd_savepoint`
/// pair for begin/commit/rollback. No vectors, no positions.
#[derive(Debug, Default)]
pub struct PositionCountingHook {
    coeff_sign: usize,
    coeff_suffix: usize,
    mvd_sign: usize,
    mvd_suffix: usize,
    /// Phase 6F.2 begin/commit/rollback savepoint (mirror of
    /// PositionLoggerHook's mvd_savepoint). `Some((sign, suffix))`
    /// snapshot taken at `begin_mvd_for_mb`; rollback truncates
    /// counters back to it, commit clears the savepoint.
    mvd_savepoint: Option<(usize, usize)>,
}

impl PositionCountingHook {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot current running totals as
    /// `[coeff_sign, coeff_suffix, mvd_sign, mvd_suffix]`. Used by
    /// the orchestrator driver to build per-GOP count rows.
    pub fn snapshot(&self) -> [usize; 4] {
        [self.coeff_sign, self.coeff_suffix, self.mvd_sign, self.mvd_suffix]
    }
}

impl StegoMbHook for PositionCountingHook {
    fn on_residual_block(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        scan_coeffs: &mut [i32],
        start_idx: usize,
        end_idx: usize,
        path_kind: ResidualPathKind,
    ) {
        use super::{
            enumerate_coeff_sign_positions, enumerate_coeff_suffix_lsb_positions,
        };
        // CoeffSignBypass count.
        let positions = enumerate_coeff_sign_positions(
            scan_coeffs, start_idx, end_idx, frame_idx, mb_addr,
            |ci| path_kind.path(ci, BinKind::Sign),
        );
        self.coeff_sign += positions.len();

        // CoeffSuffixLsb count.
        let positions = enumerate_coeff_suffix_lsb_positions(
            scan_coeffs, start_idx, end_idx, frame_idx, mb_addr,
            |ci| path_kind.path(ci, BinKind::SuffixLsb),
        );
        self.coeff_suffix += positions.len();
    }

    fn on_mvd_slot(
        &mut self,
        frame_idx: u32,
        mb_addr: u32,
        slot: &mut MvdSlot,
    ) {
        use super::{
            enumerate_mvd_sign_positions, enumerate_mvd_suffix_lsb_positions,
        };
        let single = [*slot];

        // MvdSignBypass count.
        let positions = enumerate_mvd_sign_positions(&single, frame_idx, mb_addr);
        self.mvd_sign += positions.len();

        // MvdSuffixLsb count.
        let positions = enumerate_mvd_suffix_lsb_positions(&single, frame_idx, mb_addr);
        self.mvd_suffix += positions.len();
    }

    fn begin_mvd_for_mb(&mut self) {
        self.mvd_savepoint = Some((self.mvd_sign, self.mvd_suffix));
    }

    fn commit_mvd_for_mb(&mut self) {
        self.mvd_savepoint = None;
    }

    fn rollback_mvd_for_mb(&mut self) {
        if let Some((s, l)) = self.mvd_savepoint.take() {
            self.mvd_sign = s;
            self.mvd_suffix = l;
        }
    }

    fn take_counts_if_counter(&mut self) -> Option<[usize; 4]> {
        Some(self.snapshot())
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

    /// #540.2 — `on_mvd_slot` is now a NO-OP. MvdSuffixLsb flips
    /// happen via emit-time abs override
    /// ([`InjectionHook::mvd_suffix_lsb_abs_override`]), not by
    /// mutating `slot.value`. This keeps `mv_grid` clean (= no
    /// cascade), so the cascade-safety filter is no longer required
    /// for primary STC.
    ///
    /// Pre-port behavior tested mutation at gate-allowed positions;
    /// this test now verifies the no-op contract across the same
    /// scenarios.
    #[test]
    fn injection_hook_on_mvd_slot_is_noop() {
        use super::super::hook::{EmbedDomain, PositionKey, SyntaxPath, BinKind};
        let test_key = || PositionKey::new(
            0, 0, EmbedDomain::MvdSuffixLsb,
            SyntaxPath::Mvd { list: 0, partition: 0, axis: Axis::X, kind: BinKind::SuffixLsb },
        );
        let install_gate = |hook: &mut InjectionHook<ForceBit>| {
            let mut set = std::collections::HashSet::new();
            set.insert(test_key());
            hook.set_mvd_msl_safe_gate(set);
        };

        // Pre-port: would mutate -5 → -6. Post-port: no-op.
        let mut hook = InjectionHook::new(ForceBit(0));
        install_gate(&mut hook);
        let mut slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: -5 };
        hook.on_mvd_slot(0, 0, &mut slot);
        assert_eq!(slot.value, -5, "on_mvd_slot must NOT mutate slot.value (#540.2)");

        // Pre-port: would mutate -12 → -11. Post-port: no-op.
        let mut hook = InjectionHook::new(ForceBit(1));
        install_gate(&mut hook);
        let mut slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: -12 };
        hook.on_mvd_slot(0, 0, &mut slot);
        assert_eq!(slot.value, -12, "on_mvd_slot must NOT mutate slot.value (#540.2)");

        // Gate not installed: still no-op (unchanged).
        let mut hook = InjectionHook::new(ForceBit(0));
        let mut slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: -5 };
        hook.on_mvd_slot(0, 0, &mut slot);
        assert_eq!(slot.value, -5,
            "no gate installed → on_mvd_slot is no-op");
    }

    /// #540.2 — Now exercise the new wire-only emit path. With a
    /// planned MvdSuffixLsb override, `mvd_suffix_lsb_abs_override`
    /// returns the target abs (slot.value untouched).
    #[test]
    fn injection_hook_mvd_suffix_lsb_abs_override_returns_target() {
        // value=12 (abs=12, walker=1, plan walker=0 from ForceBit(0)).
        // walker_bit != plan_bit → flip. flipped_magnitude(12) at
        // threshold 9: 12 != 9 → abs-1 = 11. Returns Some(11).
        let mut hook = InjectionHook::new(ForceBit(0));
        let slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 12 };
        let r = hook.mvd_suffix_lsb_abs_override(0, 0, &slot);
        assert_eq!(r, Some(11),
            "plan_bit=0 (walker) on abs=12 (walker=1) → target=11");

        // value=11 (abs=11, walker=0). plan walker=0 → no flip.
        let mut hook = InjectionHook::new(ForceBit(0));
        let slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 11 };
        let r = hook.mvd_suffix_lsb_abs_override(0, 0, &slot);
        assert_eq!(r, None, "walker_bit == plan_bit → no override");

        // abs=9 boundary: walker=0, plan=1 → flip. flipped_magnitude(9)
        // at threshold 9 → abs+1 = 10. Returns Some(10).
        let mut hook = InjectionHook::new(ForceBit(1));
        let slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 9 };
        let r = hook.mvd_suffix_lsb_abs_override(0, 0, &slot);
        assert_eq!(r, Some(10),
            "boundary abs=9 must flip +1 to keep target in emit range");

        // abs < threshold → no override (no suffix bin emitted).
        let mut hook = InjectionHook::new(ForceBit(1));
        let slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 5 };
        let r = hook.mvd_suffix_lsb_abs_override(0, 0, &slot);
        assert_eq!(r, None, "abs < threshold(9) → no override");

        // value=0 → no override.
        let mut hook = InjectionHook::new(ForceBit(1));
        let slot = MvdSlot { list: 0, partition: 0, axis: Axis::X, value: 0 };
        let r = hook.mvd_suffix_lsb_abs_override(0, 0, &slot);
        assert_eq!(r, None, "zero MVD → no override");
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
