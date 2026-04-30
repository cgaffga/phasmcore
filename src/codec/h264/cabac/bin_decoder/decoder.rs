// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Top-level CABAC bin-level decoder: engine + contexts + neighbors.
//
// Mirrors `cabac::CabacEncoder` — same 1024-context table, same
// neighbor row buffer, opposite arithmetic direction. The per-syntax
// decoders in [`super::syntax`] operate on this struct, mirroring
// `cabac::encoder` fn-for-fn.

use crate::codec::h264::cabac::context::{initialize_contexts, CabacContext, CabacInitSlot};
use crate::codec::h264::cabac::neighbor::CabacNeighborContext;

use super::engine::{CabacDecodeEngine, DecodeError};

/// Top-level CABAC bin decoder. Holds the arithmetic engine, the
/// 1024 per-ctxIdx contexts, and the neighbor row buffer used by
/// per-bin ctx_idx_inc derivations.
///
/// Constructed at slice start with `new_slice`. Each per-syntax-element
/// decoder in [`super::syntax`] takes a `&mut CabacDecoder` and
/// returns the decoded value (mirror of encoder pattern).
pub struct CabacDecoder<'a> {
    pub engine: CabacDecodeEngine<'a>,
    pub contexts: Box<[CabacContext; 1024]>,
    pub neighbors: CabacNeighborContext,
}

impl<'a> CabacDecoder<'a> {
    /// Construct a fresh decoder at slice start. Initializes all
    /// 1024 contexts from the spec § 9.3.1.1 init formula using the
    /// provided slot (I/SI or P/B init_idc) and slice QP. Reads the
    /// initial 9 bits of the byte stream into `cod_i_offset`.
    pub fn new_slice(
        bytes: &'a [u8],
        slot: CabacInitSlot,
        slice_qp_y: i32,
        mb_width: usize,
    ) -> Result<Self, DecodeError> {
        let contexts = Box::new(initialize_contexts(slot, slice_qp_y));
        let neighbors = CabacNeighborContext::new(mb_width, slot);
        let engine = CabacDecodeEngine::new(bytes)?;
        Ok(Self { engine, contexts, neighbors })
    }

    /// Decode one regular bin at the given `ctx_idx`. Mirrors the
    /// encoder's `encode_dec`.
    #[inline]
    pub(crate) fn decode_dec(&mut self, ctx_idx: u32) -> Result<u8, DecodeError> {
        let ctx = &mut self.contexts[ctx_idx as usize];
        self.engine.decode_decision_with_ctx_idx(ctx, ctx_idx)
    }

    /// Decode one bypass (equal-probability) bin.
    #[inline]
    pub fn decode_bypass(&mut self) -> Result<u8, DecodeError> {
        self.engine.decode_bypass()
    }

    /// Decode the special terminating bin (`end_of_slice_flag` /
    /// I_PCM indicator). Returns 1 when the slice ends after the
    /// current MB.
    #[inline]
    pub fn decode_terminate(&mut self) -> Result<u8, DecodeError> {
        self.engine.decode_terminate()
    }

    /// Bin count diagnostic (mirror of encoder's `bin_count`).
    #[inline]
    pub fn bin_count(&self) -> u32 {
        self.engine.bin_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::cabac::CabacEncoder;

    #[test]
    fn new_slice_initializes_1024_contexts() {
        // Encode a minimal slice (terminate-only) to get bytes.
        let mut enc = CabacEncoder::new_slice(CabacInitSlot::ISI, 26, 4);
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();

        let dec = CabacDecoder::new_slice(&bytes, CabacInitSlot::ISI, 26, 4)
            .expect("init");
        assert_eq!(dec.contexts.len(), 1024);
        // ctxIdx 276 is always non-adapting (pStateIdx=63, valMPS=0).
        assert_eq!(dec.contexts[276].p_state_idx(), 63);
        assert_eq!(dec.contexts[276].val_mps(), 0);
    }

    #[test]
    fn p_slice_initializes_with_correct_slot() {
        // Capture encoder context table before finish() consumes enc.
        let mut enc = CabacEncoder::new_slice(CabacInitSlot::PIdc1, 30, 8);
        let enc_ctx_snapshot: Vec<(u8, u8)> = enc
            .contexts
            .iter()
            .map(|c| (c.p_state_idx(), c.val_mps()))
            .collect();
        enc.engine.encode_terminate(1);
        let bytes = enc.finish();
        let dec = CabacDecoder::new_slice(&bytes, CabacInitSlot::PIdc1, 30, 8)
            .expect("init");
        // The encoder + decoder must have agreed on the initial
        // context table because they used the same slot+QP.
        for i in 0..1024 {
            assert_eq!(
                (dec.contexts[i].p_state_idx(), dec.contexts[i].val_mps()),
                enc_ctx_snapshot[i],
                "ctxIdx {i} initial state mismatch",
            );
        }
    }
}
