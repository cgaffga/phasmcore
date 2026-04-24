// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! H.264 CABAC arithmetic engine. Phase 6C.1.
//!
//! Implements the encoder side of spec § 9.3.4. Takes bins +
//! contexts; produces a byte stream suitable for slice-data RBSP.
//!
//! Implementation style: byte-buffering (HM-family). Matches the
//! approach used by our HEVC CABAC encoder to keep the carry
//! propagation at byte granularity — bit-level PutBit cost the HEVC
//! encoder weeks of debug; we don't repeat that mistake.
//!
//! Spec references:
//!  - § 9.3.4.1 engine state + initial state (eq 9-5 et seq).
//!  - § 9.3.4.2 EncodeDecision (regular bins, eq 9-25 to eq 9-28).
//!  - § 9.3.4.3 RenormE + PutBit (Figures 9-8, 9-9).
//!  - § 9.3.4.4 EncodeBypass (Figure 9-10).
//!  - § 9.3.4.5 EncodeTerminate + EncodeFlush (Figures 9-11, 9-12).
//!
//! Algorithm note: `docs/design/h264-encoder-algorithms/cabac-engine.md`.

use super::context::CabacContext;
use super::tables::RANGE_TAB_LPS;

/// CABAC arithmetic encoder state + output buffer.
///
/// Per spec § 9.3.4.1, five registers drive the engine:
/// `codIRange`, `codILow`, `firstBitFlag`, `bitsOutstanding`,
/// `binCountsInNALunits`. The byte-buffering implementation
/// collapses `firstBitFlag + bitsOutstanding` into a running
/// `buffered_byte + num_buffered_bytes` pair (equivalent; see
/// note on byte-level carry propagation below).
pub struct CabacEngine {
    /// Spec `codILow`, left-justified in 32 bits.
    low: u32,
    /// Spec `codIRange`, 9-bit logical value (2..=510).
    range: u32,
    /// Bits remaining before the next byte flush. Starts at 23 (for
    /// a 32-bit `low` with 9-bit `range`: 32 − 9 = 23). Mirrors the
    /// HM naming.
    bits_left: i32,
    /// Pending output byte whose final value is not yet known
    /// (because a future MPS/LPS might carry into it). Only emitted
    /// once we see a non-carry-ambiguous byte.
    buffered_byte: u8,
    /// Number of 0xFF bytes buffered behind `buffered_byte`
    /// (pending-carry chain — spec's `bitsOutstanding` in byte form).
    num_buffered_bytes: u32,
    /// Emitted output bytes.
    output: Vec<u8>,
    /// Spec `binCountsInNALunits`. Incremented on every bin emit.
    /// Persists across slices within a picture (§ 9.3.4.1).
    bin_counts: u32,
    /// Optional encoder-side bin trace for diff'ing against the
    /// spec-direct decoder (`examples/h264_cabac_decoder.rs`). When
    /// `Some`, every `encode_decision` / `encode_bypass` /
    /// `encode_terminate` call appends a line in the same format the
    /// decoder emits. Enable via `CabacEncoder::set_label(...)` +
    /// calling `take_trace()` at slice end.
    pub trace: Option<Vec<String>>,
    /// Current label shown in trace lines (set by per-syntax-element
    /// encoders to describe what bins are being emitted).
    pub trace_label: String,
}

impl CabacEngine {
    /// Construct a new engine in the spec's initial state
    /// (§ 9.3.4.1, invoked by `InitializeDecodingEngine`).
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 510,
            bits_left: 23,
            buffered_byte: 0xFF,
            num_buffered_bytes: 0,
            output: Vec::new(),
            bin_counts: 0,
            trace: None,
            trace_label: String::new(),
        }
    }

    /// Number of bins encoded so far (`binCountsInNALunits`). Drives
    /// the `cabac_zero_word` byte-stuffing formula (§ 9.3.4.6).
    pub fn bin_count(&self) -> u32 {
        self.bin_counts
    }

    /// Encode a regular bin with its context (spec § 9.3.4.2). The
    /// context is updated per `transIdxLPS` / `transIdxMPS` depending
    /// on whether the bin was the MPS or LPS.
    #[inline]
    pub fn encode_decision(&mut self, bin: u8, ctx: &mut CabacContext) {
        self.encode_decision_with_ctx_idx(bin, ctx, u32::MAX);
    }

    /// Variant that also carries a `ctx_idx` for trace logging. The
    /// per-syntax-element encoders know the absolute ctxIdx; they
    /// pass it through here so the trace line matches the decoder's
    /// format.
    #[inline]
    pub fn encode_decision_with_ctx_idx(
        &mut self,
        bin: u8,
        ctx: &mut CabacContext,
        ctx_idx: u32,
    ) {
        let (pre_range, pre_low, pre_state, pre_mps) = (
            self.range, self.low, ctx.p_state_idx(), ctx.val_mps(),
        );
        let p_state = ctx.p_state_idx() as usize;
        let q_idx = ((self.range >> 6) & 3) as usize;
        let range_lps = RANGE_TAB_LPS[p_state][q_idx] as u32;
        self.range -= range_lps;

        if bin != ctx.val_mps() {
            self.low += self.range;
            self.range = range_lps;
            ctx.update_lps();
        } else {
            ctx.update_mps();
        }

        self.renormalize();
        self.bin_counts += 1;
        if let Some(tr) = self.trace.as_mut() {
            tr.push(format!(
                "ENC {}: ctx={} pre_range=0x{:x} pre_low=0x{:x} p_state_pre={} val_mps_pre={} \
                 bin={} post_range=0x{:x} post_low=0x{:x} post_state={} post_mps={}",
                self.trace_label, ctx_idx, pre_range, pre_low, pre_state, pre_mps, bin,
                self.range, self.low, ctx.p_state_idx(), ctx.val_mps(),
            ));
            let _ = (pre_range, pre_low, pre_state, pre_mps);
        }
    }

    /// Encode a bypass bin (equal-probability, no context;
    /// spec § 9.3.4.4).
    #[inline]
    pub fn encode_bypass(&mut self, bin: u8) {
        let pre_low = self.low;
        self.low <<= 1;
        if bin != 0 {
            self.low = self.low.wrapping_add(self.range);
        }
        self.bits_left -= 1;
        if self.bits_left < 12 {
            self.write_out();
        }
        self.bin_counts += 1;
        if let Some(tr) = self.trace.as_mut() {
            tr.push(format!(
                "ENC {}: BYPASS pre_low=0x{:x} bin={} post_low=0x{:x}",
                self.trace_label, pre_low, bin, self.low,
            ));
        }
    }

    /// Encode a terminating bin (`end_of_slice_flag` or the I_PCM
    /// indicator bit of `mb_type`; spec § 9.3.4.5). When `bin == 1`,
    /// this also performs the final flush (`EncodeFlush`).
    pub fn encode_terminate(&mut self, bin: u8) {
        let pre_range = self.range;
        let pre_low = self.low;
        self.range -= 2;
        if bin != 0 {
            // Final flush path (Figure 9-12). Set range=2<<7, shift
            // low by 7 bits, and let write_out emit them.
            self.low = self.low.wrapping_add(self.range);
            self.low <<= 7;
            self.range = 2 << 7;
            self.bits_left -= 7;
            if self.bits_left < 12 {
                self.write_out();
            }
        } else if self.range < 256 {
            // Non-final terminate with range < 256 — one renorm step
            // (never more, because `range - 2` at the smallest
            // possible post-decision value still has at most one
            // missing bit of precision).
            self.low <<= 1;
            self.range <<= 1;
            self.bits_left -= 1;
            if self.bits_left < 12 {
                self.write_out();
            }
        }
        self.bin_counts += 1;
        if let Some(tr) = self.trace.as_mut() {
            tr.push(format!(
                "ENC {}: TERMINATE pre_range=0x{:x} pre_low=0x{:x} bin={} \
                 post_range=0x{:x} post_low=0x{:x}",
                self.trace_label, pre_range, pre_low, bin, self.range, self.low,
            ));
        }
    }

    /// Finish the slice: flush remaining state into bytes and append
    /// the RBSP stop bit (spec § 9.3.4.5 Figure 9-12 + § 7.3.2.10
    /// rbsp_trailing_bits).
    ///
    /// Returns the complete RBSP bytes for the slice's CABAC portion.
    /// Caller is responsible for prepending the slice header (which
    /// is CAVLC-coded — header does not live in this engine's
    /// output) and for appending any `cabac_zero_word` stuffing per
    /// spec § 9.3.4.6 (see `cabac-validation.md`).
    pub fn finish(mut self) -> Vec<u8> {
        // The final `encode_terminate(1)` call (which every slice
        // MUST emit as `end_of_slice_flag=1`) has already pushed the
        // flush bits into `low` via the 7-shift path. Now we emit
        // the remaining bits + the RBSP stop-one-bit + alignment.

        // Step 1: check if there's a pending carry in the MSB that
        // needs to propagate through buffered 0xFF bytes.
        let carry = if self.bits_left < 32 {
            self.low >> (32 - self.bits_left as u32)
        } else {
            0
        };

        // Step 2: flush buffered bytes with the carry applied.
        if carry != 0 {
            if self.num_buffered_bytes > 0 {
                self.output.push(self.buffered_byte.wrapping_add(1));
                // Carry turned pending 0xFF bytes into 0x00 bytes.
                for _ in 1..self.num_buffered_bytes {
                    self.output.push(0x00);
                }
            }
            self.low -= 1u32 << (32 - self.bits_left as u32);
        } else {
            if self.num_buffered_bytes > 0 {
                self.output.push(self.buffered_byte);
            }
            for _ in 1..self.num_buffered_bytes {
                self.output.push(0xFF);
            }
        }

        // Step 3: emit the remaining CABAC bits from `low`. Per
        // HM / spec Figure 9-12, the encoder writes (24 - bits_left)
        // bits of (low >> 8) to the output, MSB-first.
        let cabac_bits = (24i32 - self.bits_left).max(0) as u32;
        let value = self.low >> 8;

        // Accumulate CABAC remaining bits + RBSP stop bit (1) +
        // alignment zeros into full bytes.
        let mut acc: u32 = 0;
        let mut acc_bits: u32 = 0;

        for i in (0..cabac_bits).rev() {
            acc = (acc << 1) | ((value >> i) & 1);
            acc_bits += 1;
            if acc_bits == 8 {
                self.output.push(acc as u8);
                acc = 0;
                acc_bits = 0;
            }
        }

        // RBSP stop-one-bit.
        acc = (acc << 1) | 1;
        acc_bits += 1;
        if acc_bits == 8 {
            self.output.push(acc as u8);
            acc = 0;
            acc_bits = 0;
        }

        // Byte-align with trailing zeros.
        if acc_bits > 0 {
            acc <<= 8 - acc_bits;
            self.output.push(acc as u8);
        }

        self.output
    }

    /// Renormalize: shift out MSBs of `low` until `range >= 256`
    /// (spec § 9.3.4.3 Figure 9-8).
    #[inline]
    fn renormalize(&mut self) {
        while self.range < 256 {
            self.low <<= 1;
            self.range <<= 1;
            self.bits_left -= 1;
        }
        if self.bits_left < 12 {
            self.write_out();
        }
    }

    /// Extract the leading byte from `low` and handle carry
    /// propagation through buffered 0xFF bytes. Equivalent to the
    /// spec's `PutBit` loop over bit-by-bit carry — but operating on
    /// whole bytes.
    fn write_out(&mut self) {
        let lead_byte = (self.low >> (24 - self.bits_left as u32)) & 0x1FF;
        self.bits_left += 8;
        self.low &= 0xFFFFFFFFu32 >> self.bits_left as u32;

        if lead_byte == 0xFF {
            // The leading byte is 0xFF → defer emission. A future
            // carry could turn it into 0x00.
            self.num_buffered_bytes += 1;
        } else {
            // A non-0xFF byte resolves any pending carry. Flush the
            // buffer chain with the appropriate carry correction.
            if self.num_buffered_bytes > 0 {
                let carry = (lead_byte >> 8) as u8;
                self.output.push(self.buffered_byte.wrapping_add(carry));
                let fill = if carry != 0 { 0x00u8 } else { 0xFFu8 };
                for _ in 1..self.num_buffered_bytes {
                    self.output.push(fill);
                }
            }
            self.num_buffered_bytes = 1;
            self.buffered_byte = (lead_byte & 0xFF) as u8;
        }
    }
}

impl Default for CabacEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_matches_spec() {
        let eng = CabacEngine::new();
        assert_eq!(eng.range, 510);
        assert_eq!(eng.low, 0);
        assert_eq!(eng.bin_counts, 0);
        assert_eq!(eng.bits_left, 23);
    }

    #[test]
    fn bin_count_advances_per_encode() {
        let mut eng = CabacEngine::new();
        let mut ctx = CabacContext::new(30, 0);
        eng.encode_decision(0, &mut ctx);
        eng.encode_bypass(1);
        eng.encode_terminate(0);
        assert_eq!(eng.bin_count(), 3);
    }

    #[test]
    fn encode_mps_run_then_terminate_produces_bytes() {
        // Biased stream: encode 100 MPS bins with a strongly-biased
        // context (pStateIdx=60 → MPS probability ≈ 0.97). Verify
        // the output compresses vs the 100-bit raw size.
        let mut eng = CabacEngine::new();
        let mut ctx = CabacContext::new(60, 0);
        for _ in 0..100 {
            eng.encode_decision(0, &mut ctx);
        }
        eng.encode_terminate(1);
        let bytes = eng.finish();
        // 100 bins → 12.5 bytes raw. CABAC compresses biased
        // content; output should be well under 12 bytes of body.
        assert!(
            bytes.len() < 12,
            "expected compression below 12 bytes, got {}",
            bytes.len()
        );
    }

    #[test]
    fn encode_random_bins_then_terminate_produces_bytes() {
        // Alternate bin values. With pStateIdx=0 (max uncertainty)
        // and valMPS flipping each LPS at the saturated state, the
        // compression ratio is near 1.0 (no compression possible on
        // random data).
        let mut eng = CabacEngine::new();
        let mut ctx = CabacContext::new(0, 0);
        for i in 0..64 {
            eng.encode_decision((i & 1) as u8, &mut ctx);
        }
        eng.encode_terminate(1);
        let bytes = eng.finish();
        assert!(!bytes.is_empty());
        // Last byte must have the RBSP stop bit somewhere in it.
        assert_ne!(bytes.last().copied(), Some(0));
    }

    #[test]
    fn bypass_bins_extend_output_linearly() {
        // Bypass bins bypass context adaptation but still count
        // toward output length. Emit 16 bypass bins + terminate.
        let mut eng = CabacEngine::new();
        for i in 0..16 {
            eng.encode_bypass((i & 1) as u8);
        }
        eng.encode_terminate(1);
        let bytes = eng.finish();
        // ≥ 2 bytes (16 bypass bins + trailing bits).
        assert!(bytes.len() >= 2);
    }

    #[test]
    fn terminate_zero_then_terminate_one_flushes_properly() {
        // Realistic multi-MB pattern: every non-last MB calls
        // encode_terminate(0); the last calls encode_terminate(1).
        // Verify: 3 non-final terminates + 1 final → valid output.
        let mut eng = CabacEngine::new();
        let mut ctx = CabacContext::new(20, 0);
        for _ in 0..3 {
            eng.encode_decision(0, &mut ctx);
            eng.encode_terminate(0);
        }
        eng.encode_decision(1, &mut ctx);
        eng.encode_terminate(1);
        let bytes = eng.finish();
        // Must emit at least 1 byte and end with a non-zero stop
        // byte (RBSP trailing 1-bit).
        assert!(!bytes.is_empty());
        assert_ne!(*bytes.last().unwrap(), 0);
    }

    #[test]
    fn single_bin_roundtrip_via_spec_decoder_pseudocode() {
        // Reference decoder that reads bins from a CABAC byte stream
        // via the spec § 9.3.3.2 pseudocode. We implement just
        // enough to verify our encoder produces decodable output.
        //
        // Test: encode 5 bins, decode them, assert round-trip.
        let bins_in = [0u8, 1, 1, 0, 1];
        let mut eng = CabacEngine::new();
        let mut ctx = CabacContext::new(20, 0);
        for &b in &bins_in {
            eng.encode_decision(b, &mut ctx);
        }
        eng.encode_terminate(1);
        let bytes = eng.finish();

        // Decode: replay the spec pseudocode.
        let mut dec = TestDecoder::new(&bytes);
        let mut ctx = CabacContext::new(20, 0);
        let bins_out: Vec<u8> = (0..5).map(|_| dec.decode_decision(&mut ctx)).collect();
        assert_eq!(bins_out, bins_in);
    }

    /// Minimal spec-pseudocode CABAC decoder for round-trip testing.
    /// Spec § 9.3.3.2 (InitializeDecodingEngine + DecodeDecision +
    /// RenormD). Only needed for unit tests.
    struct TestDecoder<'a> {
        bytes: &'a [u8],
        cod_i_offset: u32,
        cod_i_range: u32,
        byte_idx: usize,
        bits_in_byte: u32,
        bit_ptr: u32,
    }

    impl<'a> TestDecoder<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            let mut d = Self {
                bytes,
                cod_i_offset: 0,
                cod_i_range: 510,
                byte_idx: 0,
                bits_in_byte: 0,
                bit_ptr: 0,
            };
            // Init: read 9 bits into codIOffset (spec
            // InitializeDecodingEngine, Figure 9-4).
            for _ in 0..9 {
                let b = d.read_bit();
                d.cod_i_offset = (d.cod_i_offset << 1) | b;
            }
            d
        }

        fn read_bit(&mut self) -> u32 {
            if self.byte_idx >= self.bytes.len() {
                return 0;
            }
            let byte = self.bytes[self.byte_idx];
            let bit = (byte >> (7 - self.bit_ptr)) & 1;
            self.bit_ptr += 1;
            if self.bit_ptr == 8 {
                self.bit_ptr = 0;
                self.byte_idx += 1;
            }
            bit as u32
        }

        fn decode_decision(&mut self, ctx: &mut CabacContext) -> u8 {
            // Spec Figure 9-6.
            let p_state = ctx.p_state_idx() as usize;
            let q_idx = ((self.cod_i_range >> 6) & 3) as usize;
            let range_lps = RANGE_TAB_LPS[p_state][q_idx] as u32;
            self.cod_i_range -= range_lps;
            let bin = if self.cod_i_offset >= self.cod_i_range {
                self.cod_i_offset -= self.cod_i_range;
                self.cod_i_range = range_lps;
                let b = 1 ^ ctx.val_mps();
                ctx.update_lps();
                b
            } else {
                let b = ctx.val_mps();
                ctx.update_mps();
                b
            };
            // RenormD.
            while self.cod_i_range < 256 {
                self.cod_i_range <<= 1;
                self.cod_i_offset = (self.cod_i_offset << 1) | self.read_bit();
            }
            self.bits_in_byte = self.bits_in_byte.wrapping_add(0); // silence unused
            bin
        }
    }
}
