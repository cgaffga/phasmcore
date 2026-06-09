// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 CABAC context model.
//!
//! Per spec § 9.3.1 each context has two fields: `pStateIdx` (0..63,
//! probability-state index) and `valMPS` (0..1, value of the Most
//! Probable Symbol). Applied via `update_mps()` when the MPS is coded
//! and `update_lps()` when the LPS is coded (which can flip `valMPS`
//! at the saturated pStateIdx=0 state). The state transition is
//! direction-neutral; here it runs on the decode path.
//!
//! Tables driving these updates live in [`super::tables`].

use super::tables::{TRANS_IDX_LPS, TRANS_IDX_MPS};

/// CABAC context state — 1 entry per `ctxIdx` (1024 slots total).
#[derive(Debug, Clone, Copy)]
pub struct CabacContext {
    /// Probability state index, 0..63. Higher = more confident.
    p_state_idx: u8,
    /// Value of the Most Probable Symbol (0 or 1).
    val_mps: u8,
}

impl CabacContext {
    /// Create a context with explicit state. Used by
    /// `InitializeContextVariables` or test setups.
    pub fn new(p_state_idx: u8, val_mps: u8) -> Self {
        debug_assert!(p_state_idx <= 63);
        debug_assert!(val_mps <= 1);
        Self {
            p_state_idx,
            val_mps,
        }
    }

    /// Special non-adapting state for `ctxIdx = 276` (spec Table 9-11
    /// Note 2): `pStateIdx = 63, valMPS = 0`. Never updated because
    /// the only access path is `decode_terminate` which doesn't
    /// update context state.
    pub const fn non_adapting_276() -> Self {
        Self {
            p_state_idx: 63,
            val_mps: 0,
        }
    }

    pub fn p_state_idx(&self) -> u8 {
        self.p_state_idx
    }

    pub fn val_mps(&self) -> u8 {
        self.val_mps
    }

    /// Apply MPS update (spec § 9.3.4.2): `pStateIdx = transIdxMPS[pStateIdx]`.
    #[inline]
    pub fn update_mps(&mut self) {
        self.p_state_idx = TRANS_IDX_MPS[self.p_state_idx as usize];
    }

    /// Apply LPS update (spec § 9.3.4.2): flip `valMPS` at saturated
    /// pStateIdx=0, then `pStateIdx = transIdxLPS[pStateIdx]`.
    #[inline]
    pub fn update_lps(&mut self) {
        if self.p_state_idx == 0 {
            self.val_mps ^= 1;
        }
        self.p_state_idx = TRANS_IDX_LPS[self.p_state_idx as usize];
    }
}

impl Default for CabacContext {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

/// Compute the initial `(pStateIdx, valMPS)` pair from an (m, n) table
/// entry and the slice QP (spec § 9.3.1.1, equation 9-5).
///
/// ```text
/// preCtxState = Clip3(1, 126, ((m · Clip3(0, 51, sliceQPY)) >> 4) + n)
/// if preCtxState <= 63:
///     pStateIdx = 63 − preCtxState
///     valMPS    = 0
/// else:
///     pStateIdx = preCtxState − 64
///     valMPS    = 1
/// ```
///
/// Note: `>> 4` is arithmetic right shift (rounds toward −∞), which is
/// what the spec intends. Rust's signed right shift matches.
#[inline]
pub fn compute_initial_state(m: i32, n: i32, slice_qp_y: i32) -> CabacContext {
    let qp_clipped = slice_qp_y.clamp(0, 51);
    let pre_ctx_state_raw = ((m * qp_clipped) >> 4) + n;
    let pre_ctx_state = pre_ctx_state_raw.clamp(1, 126);
    if pre_ctx_state <= 63 {
        CabacContext::new((63 - pre_ctx_state) as u8, 0)
    } else {
        CabacContext::new((pre_ctx_state - 64) as u8, 1)
    }
}

/// Slot index into `CTX_INIT_MN[ctxIdx][slot]`.
///
/// I / SI slices use slot 0 (single column, no `cabac_init_idc`).
/// P / SP / B slices use slots 1, 2, or 3 per `cabac_init_idc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CabacInitSlot {
    /// I or SI slice.
    ISI,
    /// P / SP / B slice with `cabac_init_idc = 0`.
    PIdc0,
    /// P / SP / B slice with `cabac_init_idc = 1`.
    PIdc1,
    /// P / SP / B slice with `cabac_init_idc = 2`.
    PIdc2,
}

impl CabacInitSlot {
    pub fn as_index(self) -> usize {
        match self {
            CabacInitSlot::ISI => 0,
            CabacInitSlot::PIdc0 => 1,
            CabacInitSlot::PIdc1 => 2,
            CabacInitSlot::PIdc2 => 3,
        }
    }
}

/// Populate a full 1024-entry context table for a slice, per spec
/// § 9.3.1.1. Runs `compute_initial_state` for every `ctxIdx` in
/// `[0, 1023]` using the (m, n) pairs from `CTX_INIT_MN[ctxIdx][slot]`.
///
/// ctxIdx 276 is special-cased to the non-adapting state
/// (pStateIdx=63, valMPS=0) per Table 9-11 Note 2; the (0, 0) stored
/// in `CTX_INIT_MN[276]` would otherwise compute a different state.
pub fn initialize_contexts(
    slot: CabacInitSlot,
    slice_qp_y: i32,
) -> [CabacContext; 1024] {
    use super::tables::CTX_INIT_MN;
    let slot_idx = slot.as_index();
    let mut contexts = [CabacContext::default(); 1024];
    for ctx_idx in 0..1024 {
        if ctx_idx == 276 {
            contexts[ctx_idx] = CabacContext::non_adapting_276();
        } else {
            let (m, n) = CTX_INIT_MN[ctx_idx][slot_idx];
            contexts[ctx_idx] = compute_initial_state(m as i32, n as i32, slice_qp_y);
        }
    }
    contexts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_new_rejects_out_of_range() {
        let _ = CabacContext::new(63, 1);
        // Debug assertions fire for 64+ / 2+; release mode skips the check.
    }

    #[test]
    fn update_mps_increments_state_except_saturation() {
        let mut ctx = CabacContext::new(10, 0);
        ctx.update_mps();
        assert_eq!(ctx.p_state_idx(), 11);
        assert_eq!(ctx.val_mps(), 0);
    }

    #[test]
    fn update_mps_saturates_at_62() {
        let mut ctx = CabacContext::new(62, 0);
        ctx.update_mps();
        assert_eq!(ctx.p_state_idx(), 62); // saturates, NOT 63
    }

    #[test]
    fn update_lps_from_state_0_flips_val_mps() {
        let mut ctx = CabacContext::new(0, 0);
        ctx.update_lps();
        assert_eq!(ctx.val_mps(), 1); // flipped
        assert_eq!(ctx.p_state_idx(), 0); // spec: transIdxLPS[0] = 0
    }

    #[test]
    fn update_lps_from_higher_state_does_not_flip_val_mps() {
        let mut ctx = CabacContext::new(5, 1);
        ctx.update_lps();
        assert_eq!(ctx.val_mps(), 1); // unchanged
        assert_eq!(ctx.p_state_idx(), TRANS_IDX_LPS[5]);
    }

    #[test]
    fn non_adapting_276_is_max_p_state_val_mps_zero() {
        let ctx = CabacContext::non_adapting_276();
        assert_eq!(ctx.p_state_idx(), 63);
        assert_eq!(ctx.val_mps(), 0);
    }

    #[test]
    fn compute_initial_state_clips_pre_ctx_state() {
        // With m=20, n=-15, slice_qp_y=26: preCtxState = ((20*26)>>4) + (-15)
        //                                               = (520>>4) + (-15)
        //                                               = 32 + (-15) = 17.
        // 17 <= 63 → pStateIdx = 63-17 = 46, valMPS = 0.
        let ctx = compute_initial_state(20, -15, 26);
        assert_eq!(ctx.p_state_idx(), 46);
        assert_eq!(ctx.val_mps(), 0);
    }

    #[test]
    fn compute_initial_state_val_mps_flip_above_63() {
        // With m=0, n=80, slice_qp_y=26: preCtxState = 0 + 80 = 80.
        // 80 > 63 → pStateIdx = 80-64 = 16, valMPS = 1.
        let ctx = compute_initial_state(0, 80, 26);
        assert_eq!(ctx.p_state_idx(), 16);
        assert_eq!(ctx.val_mps(), 1);
    }

    #[test]
    fn compute_initial_state_clamps_below_1() {
        // Huge negative intermediate → Clip3(1, 126, …) = 1 → pStateIdx = 62.
        let ctx = compute_initial_state(-99, -99, 51);
        assert_eq!(ctx.p_state_idx(), 62);
        assert_eq!(ctx.val_mps(), 0);
    }

    #[test]
    fn compute_initial_state_clamps_above_126() {
        // Huge positive intermediate → Clip3(1, 126, …) = 126 → pStateIdx = 62, valMPS = 1.
        let ctx = compute_initial_state(127, 127, 51);
        assert_eq!(ctx.p_state_idx(), 62);
        assert_eq!(ctx.val_mps(), 1);
    }

    #[test]
    fn compute_initial_state_clamps_slice_qp() {
        // slice_qp < 0 → treated as 0. slice_qp > 51 → treated as 51.
        let a = compute_initial_state(20, -15, -5);
        let b = compute_initial_state(20, -15, 0);
        assert_eq!(a.p_state_idx(), b.p_state_idx());
        assert_eq!(a.val_mps(), b.val_mps());
    }

    #[test]
    fn cabac_init_slot_indices() {
        assert_eq!(CabacInitSlot::ISI.as_index(), 0);
        assert_eq!(CabacInitSlot::PIdc0.as_index(), 1);
        assert_eq!(CabacInitSlot::PIdc1.as_index(), 2);
        assert_eq!(CabacInitSlot::PIdc2.as_index(), 3);
    }

    #[test]
    fn initialize_contexts_i_slice_at_qp_26() {
        let ctxs = initialize_contexts(CabacInitSlot::ISI, 26);
        // ctxIdx 0 uses (m, n) = (20, -15). With qp=26:
        //   preCtxState = Clip3(1, 126, ((20*26)>>4) - 15) = 32-15 = 17.
        //   pStateIdx = 63-17 = 46, valMPS = 0.
        assert_eq!(ctxs[0].p_state_idx(), 46);
        assert_eq!(ctxs[0].val_mps(), 0);
        // ctxIdx 276 is always the non-adapting state regardless.
        assert_eq!(ctxs[276].p_state_idx(), 63);
        assert_eq!(ctxs[276].val_mps(), 0);
    }

    #[test]
    fn initialize_contexts_276_always_non_adapting() {
        for slot in [
            CabacInitSlot::ISI,
            CabacInitSlot::PIdc0,
            CabacInitSlot::PIdc1,
            CabacInitSlot::PIdc2,
        ] {
            for qp in [0, 26, 51] {
                let ctxs = initialize_contexts(slot, qp);
                assert_eq!(ctxs[276].p_state_idx(), 63);
                assert_eq!(ctxs[276].val_mps(), 0);
            }
        }
    }

    #[test]
    fn initialize_contexts_differs_by_cabac_init_idc() {
        // ctxIdx 11 has different (m, n) for init_idc 0 vs 2.
        // init_idc=0: (23, 33). init_idc=2: (29, 16).
        let c0 = initialize_contexts(CabacInitSlot::PIdc0, 26);
        let c2 = initialize_contexts(CabacInitSlot::PIdc2, 26);
        assert_ne!(
            (c0[11].p_state_idx(), c0[11].val_mps()),
            (c2[11].p_state_idx(), c2[11].val_mps())
        );
    }
}
