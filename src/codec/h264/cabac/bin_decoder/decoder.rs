// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Top-level CABAC bin-level decoder: engine + contexts + neighbors.
//
// Decode direction only: a full 1024-context table, a neighbor row
// buffer, and the spec arithmetic decode engine. The per-syntax
// decoders in [`super::syntax`] operate on this struct.

use crate::codec::h264::cabac::context::{initialize_contexts, CabacContext, CabacInitSlot};
use crate::codec::h264::cabac::neighbor::CabacNeighborContext;

use super::engine::{CabacDecodeEngine, DecodeError};

/// Top-level CABAC bin decoder. Holds the arithmetic engine, the
/// 1024 per-ctxIdx contexts, and the neighbor row buffer used by
/// per-bin ctx_idx_inc derivations.
///
/// Constructed at slice start with `new_slice`. Each per-syntax-element
/// decoder in [`super::syntax`] takes a `&mut CabacDecoder` and
/// returns the decoded value.
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

    /// Decode one regular bin at the given `ctx_idx`.
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

    /// Bin count diagnostic.
    #[inline]
    pub fn bin_count(&self) -> u32 {
        self.engine.bin_count()
    }

    /// RBSP bit offset of the next raw bit. See
    /// [`CabacDecodeEngine::next_rbsp_bit_offset`]. Used by the
    /// bitstream-mod stego path to capture the position of each
    /// bypass-coded stego bin.
    #[inline]
    pub fn next_rbsp_bit_offset(&self) -> u64 {
        self.engine.next_rbsp_bit_offset()
    }
}
