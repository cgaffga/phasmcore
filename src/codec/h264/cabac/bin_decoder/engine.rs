// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// H.264 CABAC bin-level decode engine (Phase 6D.2).
//
// Spec references:
//  - § 9.3.1.2 InitializeDecodingEngine (Figure 9-12)
//  - § 9.3.3.2.1 DecodeDecision (Figure 9-6)
//  - § 9.3.3.2.3 DecodeBypass (Figure 9-9)
//  - § 9.3.3.2.4 DecodeTerminate (Figure 9-10)
//  - § 9.3.3.2.2 RenormD (Figure 9-7)
//
// Implementation pattern: bit-level read_bit feeding the standard
// `cod_i_offset` / `cod_i_range` registers. Symmetric to
// `cabac::engine::CabacEngine` — same tables, same arithmetic, opposite
// direction. By construction (paired implementation) round-trip is
// exact when fed bytes from our own encoder.

use crate::codec::h264::cabac::context::CabacContext;
use crate::codec::h264::cabac::tables::RANGE_TAB_LPS;

/// Errors the decoder surfaces. All recoverable at the caller's
/// option (e.g. "ran past end of slice" can be a benign EOF in some
/// contexts; the slice walker decides).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DecodeError {
    /// Tried to read past the end of the input byte buffer. Most
    /// often indicates a malformed bitstream or end-of-slice reached
    /// without seeing the end-of-slice flag.
    UnexpectedEof,
    /// Bin decoder parsed a value the current implementation scope
    /// does not support (e.g. §6E-A6.3 walker encountering B-slice
    /// `sub_mb_type` 4..=12). The static message names the scope —
    /// the walker layer adds dynamic context when wrapping into
    /// `WalkError::H264(Unsupported)`.
    Unsupported(&'static str),
}

/// CABAC arithmetic decoder state.
///
/// Spec § 9.3.1.2 / Figure 9-12: at slice start the decoder reads 9
/// bits into `codIOffset` and sets `codIRange = 510`. The 9-bit
/// initialization makes the first bin's `(codIRange - codIRangeLPS)`
/// comparison meaningful before any `RenormD` runs.
///
/// Output of every `decode_*` method is a `u8` (0 or 1) bin, mirroring
/// the encoder's input.
pub struct CabacDecodeEngine<'a> {
    /// CABAC bytes for the current slice (post-emulation-prevention
    /// removed; same byte stream the encoder produced).
    bytes: &'a [u8],
    /// Byte index currently being consumed from `bytes`.
    byte_idx: usize,
    /// Bit pointer within `bytes[byte_idx]` (0 = MSB, 7 = LSB).
    bit_ptr: u32,

    /// Spec `codIOffset`. 9-bit window into the bitstream that
    /// `DecodeDecision` compares against `codIRange - codIRangeLPS`.
    cod_i_offset: u32,
    /// Spec `codIRange`. 9-bit logical value, 256..=510.
    cod_i_range: u32,

    /// Bins decoded so far. Diagnostic counter mirrors the encoder's
    /// `bin_counts` for trace cross-checks.
    bin_counts: u32,

    /// Optional bin trace for diff'ing against the encoder's emit
    /// trace (`CabacEngine::trace`). Same line format both sides.
    pub trace: Option<Vec<String>>,
    /// Current label for trace lines (set by per-syntax-element
    /// decoders to describe what bins are being decoded).
    pub trace_label: String,
}

impl<'a> CabacDecodeEngine<'a> {
    /// Construct a fresh decoder over `bytes` and run
    /// `InitializeDecodingEngine` (spec Figure 9-12). Returns
    /// `UnexpectedEof` if the first 9 bits cannot be read.
    pub fn new(bytes: &'a [u8]) -> Result<Self, DecodeError> {
        let mut eng = Self {
            bytes,
            byte_idx: 0,
            bit_ptr: 0,
            cod_i_offset: 0,
            cod_i_range: 510,
            bin_counts: 0,
            trace: None,
            trace_label: String::new(),
        };
        for _ in 0..9 {
            let b = eng.read_bit()?;
            eng.cod_i_offset = (eng.cod_i_offset << 1) | b;
        }
        Ok(eng)
    }

    /// Number of bins decoded so far. Diagnostic only.
    #[inline]
    pub fn bin_count(&self) -> u32 {
        self.bin_counts
    }

    /// Number of bytes consumed so far. Useful for slice-end
    /// detection when the caller knows the slice's RBSP boundary.
    #[inline]
    pub fn bytes_consumed(&self) -> usize {
        self.byte_idx + (if self.bit_ptr > 0 { 1 } else { 0 })
    }

    /// Decode one regular (context-coded) bin. Spec § 9.3.3.2.1
    /// (Figure 9-6). Updates `ctx` per `transIdxLPS` / `transIdxMPS`.
    #[inline]
    pub fn decode_decision(&mut self, ctx: &mut CabacContext) -> Result<u8, DecodeError> {
        self.decode_decision_with_ctx_idx(ctx, u32::MAX)
    }

    /// Variant that also carries a `ctx_idx` for trace logging,
    /// matching the encoder's
    /// [`CabacEngine::encode_decision_with_ctx_idx`](crate::codec::h264::cabac::engine::CabacEngine::encode_decision_with_ctx_idx).
    pub fn decode_decision_with_ctx_idx(
        &mut self,
        ctx: &mut CabacContext,
        ctx_idx: u32,
    ) -> Result<u8, DecodeError> {
        let pre_range = self.cod_i_range;
        let pre_offset = self.cod_i_offset;
        let pre_state = ctx.p_state_idx();
        let pre_mps = ctx.val_mps();

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

        self.renormalize_d()?;
        self.bin_counts += 1;

        if let Some(tr) = self.trace.as_mut() {
            tr.push(format!(
                "DEC {}: ctx={} pre_range=0x{:x} pre_offset=0x{:x} p_state_pre={} val_mps_pre={} \
                 bin={} post_range=0x{:x} post_offset=0x{:x} post_state={} post_mps={}",
                self.trace_label, ctx_idx, pre_range, pre_offset, pre_state, pre_mps, bin,
                self.cod_i_range, self.cod_i_offset, ctx.p_state_idx(), ctx.val_mps(),
            ));
        }
        Ok(bin)
    }

    /// Decode one bypass (equal-probability) bin. Spec § 9.3.3.2.3
    /// (Figure 9-9). No context; doubles `codIOffset` and reads one
    /// bit.
    #[inline]
    pub fn decode_bypass(&mut self) -> Result<u8, DecodeError> {
        let pre_offset = self.cod_i_offset;
        self.cod_i_offset = (self.cod_i_offset << 1) | self.read_bit()?;
        let bin = if self.cod_i_offset >= self.cod_i_range {
            self.cod_i_offset -= self.cod_i_range;
            1
        } else {
            0
        };
        self.bin_counts += 1;
        if let Some(tr) = self.trace.as_mut() {
            tr.push(format!(
                "DEC {}: BYPASS pre_offset=0x{:x} bin={} post_offset=0x{:x}",
                self.trace_label, pre_offset, bin, self.cod_i_offset,
            ));
        }
        Ok(bin)
    }

    /// Decode the special `end_of_slice_flag` / I_PCM-indicator
    /// terminating bin. Spec § 9.3.3.2.4 (Figure 9-10).
    ///
    /// Returns `1` for the terminating decision (slice ends after
    /// this MB; or this MB is I_PCM). On `0`, normal decoding
    /// continues.
    pub fn decode_terminate(&mut self) -> Result<u8, DecodeError> {
        let pre_range = self.cod_i_range;
        let pre_offset = self.cod_i_offset;
        self.cod_i_range -= 2;
        let bin = if self.cod_i_offset >= self.cod_i_range {
            1
        } else {
            self.renormalize_d()?;
            0
        };
        self.bin_counts += 1;
        if let Some(tr) = self.trace.as_mut() {
            tr.push(format!(
                "DEC {}: TERMINATE pre_range=0x{:x} pre_offset=0x{:x} bin={} \
                 post_range=0x{:x} post_offset=0x{:x}",
                self.trace_label, pre_range, pre_offset, bin,
                self.cod_i_range, self.cod_i_offset,
            ));
        }
        Ok(bin)
    }

    /// Read one raw bit from the byte stream (spec
    /// `read_bits( 1 )`). MSB-first within each byte.
    #[inline]
    fn read_bit(&mut self) -> Result<u32, DecodeError> {
        if self.byte_idx >= self.bytes.len() {
            // CABAC decoders are allowed to read past the end of the
            // RBSP body (spec § 9.3.3.2 Note 1) — the spec defines
            // the behaviour as "reading zero bits". We follow that
            // convention: return 0 for over-reads, NOT an error.
            //
            // The slice walker is responsible for detecting actual
            // end-of-slice via the terminating bin path.
            return Ok(0);
        }
        let byte = self.bytes[self.byte_idx];
        let bit = ((byte >> (7 - self.bit_ptr)) & 1) as u32;
        self.bit_ptr += 1;
        if self.bit_ptr == 8 {
            self.bit_ptr = 0;
            self.byte_idx += 1;
        }
        Ok(bit)
    }

    /// Renormalize: shift `cod_i_range` left until it's >= 256, with
    /// matching shifts of `cod_i_offset` and bit-pulls from the byte
    /// stream. Spec § 9.3.3.2.2 (Figure 9-7).
    #[inline]
    fn renormalize_d(&mut self) -> Result<(), DecodeError> {
        while self.cod_i_range < 256 {
            self.cod_i_range <<= 1;
            self.cod_i_offset = (self.cod_i_offset << 1) | self.read_bit()?;
        }
        Ok(())
    }
}

// ─── Helper: decode a sequence of bins from a byte buffer ──────────────
//
// Tests + downstream callers often want to decode a known number of
// bins from a byte slice without setting up the full slice walker.
// This is the symmetric of the encoder's "encode N bins → finish()"
// idiom.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::cabac::engine::CabacEngine;

    /// Helper: encode N bins via the production encoder, then decode
    /// them back via our new decoder. Round-trip property: input
    /// bins == output bins.
    fn roundtrip_decisions(bins: &[u8], p_state: u8, val_mps: u8) {
        let mut enc = CabacEngine::new();
        let mut ctx_e = CabacContext::new(p_state, val_mps);
        for &b in bins {
            enc.encode_decision(b, &mut ctx_e);
        }
        enc.encode_terminate(1);
        let bytes = enc.finish();

        let mut dec = CabacDecodeEngine::new(&bytes).expect("init");
        let mut ctx_d = CabacContext::new(p_state, val_mps);
        let out: Vec<u8> = (0..bins.len())
            .map(|_| dec.decode_decision(&mut ctx_d).expect("decode"))
            .collect();
        assert_eq!(out, bins, "regular-bin roundtrip");
    }

    #[test]
    fn engine_init_reads_9_bits() {
        // Construct a buffer where the first 9 bits are 0_1010_1010
        // (=0x0AA), bytes 0xAA 0x55 ...
        let bytes = [0b0101_0101, 0b0101_0101, 0xFF];
        let dec = CabacDecodeEngine::new(&bytes).expect("init");
        // 9 MSBs of the buffer = 0_1010_1010 = 0x0AA = 170.
        assert_eq!(dec.cod_i_offset, 0b0_0101_0101_0);
        assert_eq!(dec.cod_i_range, 510);
    }

    #[test]
    fn roundtrip_single_bin_zero() {
        roundtrip_decisions(&[0], 30, 0);
    }

    #[test]
    fn roundtrip_single_bin_one() {
        roundtrip_decisions(&[1], 30, 0);
    }

    #[test]
    fn roundtrip_alternating_bins() {
        let bins: Vec<u8> = (0..32).map(|i| (i & 1) as u8).collect();
        roundtrip_decisions(&bins, 0, 0);
    }

    #[test]
    fn roundtrip_biased_mps_run() {
        let bins = vec![0u8; 100];
        roundtrip_decisions(&bins, 60, 0);
    }

    #[test]
    fn roundtrip_random_bins() {
        // Deterministic "random": bits of a small LCG.
        let mut s: u32 = 0x1234_5678;
        let bins: Vec<u8> = (0..64)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s & 1) as u8
            })
            .collect();
        roundtrip_decisions(&bins, 20, 1);
    }

    #[test]
    fn roundtrip_mixed_decision_and_bypass() {
        // Mix: 4 regular bins + 8 bypass + 4 regular.
        let regular = [0u8, 1, 1, 0];
        let bypass = [1u8, 0, 1, 1, 0, 0, 1, 0];
        let regular2 = [1u8, 0, 1, 1];

        let mut enc = CabacEngine::new();
        let mut ctx = CabacContext::new(20, 0);
        for &b in &regular {
            enc.encode_decision(b, &mut ctx);
        }
        for &b in &bypass {
            enc.encode_bypass(b);
        }
        let mut ctx2 = CabacContext::new(20, 0);
        // Re-use ctx for regular2 (same context model that the
        // encoder advanced through the first run).
        for &b in &regular2 {
            enc.encode_decision(b, &mut ctx);
        }
        enc.encode_terminate(1);
        let bytes = enc.finish();

        let mut dec = CabacDecodeEngine::new(&bytes).expect("init");
        let mut ctx_d = CabacContext::new(20, 0);
        let out_regular: Vec<u8> = (0..regular.len())
            .map(|_| dec.decode_decision(&mut ctx_d).unwrap())
            .collect();
        assert_eq!(out_regular, regular);
        let out_bypass: Vec<u8> = (0..bypass.len())
            .map(|_| dec.decode_bypass().unwrap())
            .collect();
        assert_eq!(out_bypass, bypass);
        // ctx_d already advanced through regular[]; continue from there.
        let _ = ctx2;
        let out_regular2: Vec<u8> = (0..regular2.len())
            .map(|_| dec.decode_decision(&mut ctx_d).unwrap())
            .collect();
        assert_eq!(out_regular2, regular2);
    }

    #[test]
    fn roundtrip_terminate_zero_then_one() {
        // Encoder: 3 non-final terminates + 1 final.
        let mut enc = CabacEngine::new();
        let mut ctx_e = CabacContext::new(20, 0);
        for _ in 0..3 {
            enc.encode_decision(0, &mut ctx_e);
            enc.encode_terminate(0);
        }
        enc.encode_decision(1, &mut ctx_e);
        enc.encode_terminate(1);
        let bytes = enc.finish();

        let mut dec = CabacDecodeEngine::new(&bytes).expect("init");
        let mut ctx_d = CabacContext::new(20, 0);
        for _ in 0..3 {
            assert_eq!(dec.decode_decision(&mut ctx_d).unwrap(), 0);
            assert_eq!(dec.decode_terminate().unwrap(), 0);
        }
        assert_eq!(dec.decode_decision(&mut ctx_d).unwrap(), 1);
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }

    #[test]
    fn bin_count_matches_encoder() {
        let bins = [0u8, 1, 1, 0, 1, 0, 0, 1];
        let mut enc = CabacEngine::new();
        let mut ctx = CabacContext::new(30, 0);
        for &b in &bins {
            enc.encode_decision(b, &mut ctx);
        }
        enc.encode_bypass(1);
        enc.encode_terminate(1);
        // Capture bin count before finish() consumes the encoder.
        let enc_bin_count = enc.bin_count();
        let bytes = enc.finish();

        let mut dec = CabacDecodeEngine::new(&bytes).expect("init");
        let mut ctx_d = CabacContext::new(30, 0);
        for _ in 0..bins.len() {
            dec.decode_decision(&mut ctx_d).unwrap();
        }
        dec.decode_bypass().unwrap();
        dec.decode_terminate().unwrap();
        // Encoder: 8 decisions + 1 bypass + 1 terminate = 10 bins.
        // Decoder must mirror.
        assert_eq!(enc_bin_count, 10);
        assert_eq!(dec.bin_count(), 10);
    }

    #[test]
    fn unexpected_eof_returns_zero_per_spec() {
        // Empty buffer: spec § 9.3.3.2 Note 1 says reading past EOF
        // returns zero bits. Our decoder follows that convention.
        // (The slice walker detects real end-of-slice via the
        // terminating bin, not via byte-stream exhaustion.)
        let bytes: [u8; 0] = [];
        let dec = CabacDecodeEngine::new(&bytes);
        // 9 bits worth of zeros → cod_i_offset = 0 after init.
        let dec = dec.expect("init succeeds with all-zero reads");
        assert_eq!(dec.cod_i_offset, 0);
    }

    #[test]
    fn decode_engine_separate_state_from_encoder() {
        // Sanity: decoding doesn't mutate any shared global state.
        let bytes_a = {
            let mut enc = CabacEngine::new();
            let mut ctx = CabacContext::new(20, 0);
            for _ in 0..10 {
                enc.encode_decision(0, &mut ctx);
            }
            enc.encode_terminate(1);
            enc.finish()
        };
        let bytes_b = {
            let mut enc = CabacEngine::new();
            let mut ctx = CabacContext::new(20, 0);
            for _ in 0..10 {
                enc.encode_decision(0, &mut ctx);
            }
            enc.encode_terminate(1);
            enc.finish()
        };
        // Two independent encodes of the same content produce the same bytes.
        assert_eq!(bytes_a, bytes_b);

        // Two independent decodes are also independent.
        let mut d1 = CabacDecodeEngine::new(&bytes_a).unwrap();
        let mut d2 = CabacDecodeEngine::new(&bytes_b).unwrap();
        let mut c1 = CabacContext::new(20, 0);
        let mut c2 = CabacContext::new(20, 0);
        for _ in 0..10 {
            assert_eq!(
                d1.decode_decision(&mut c1).unwrap(),
                d2.decode_decision(&mut c2).unwrap(),
            );
        }
    }
}
