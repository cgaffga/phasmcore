// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// H.264 CABAC bin-level decode engine.
//
// Spec references:
//  - § 9.3.1.2 InitializeDecodingEngine (Figure 9-12)
//  - § 9.3.3.2.1 DecodeDecision (Figure 9-6)
//  - § 9.3.3.2.3 DecodeBypass (Figure 9-9)
//  - § 9.3.3.2.4 DecodeTerminate (Figure 9-10)
//  - § 9.3.3.2.2 RenormD (Figure 9-7)
//
// Implementation pattern: bit-level read_bit feeding the standard
// `cod_i_offset` / `cod_i_range` registers. This is the decode
// direction only — the forward CABAC encoder was removed in the
// video-retirement; production stego bytes come off the OpenH264 fork
// at emit. The walker decodes that stream using the same spec tables
// and arithmetic, opposite direction, and reads back the bins
// embedded at the bypass-bin emit site exactly.

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
    /// does not support (e.g. the walker encountering B-slice
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
/// Output of every `decode_*` method is a `u8` (0 or 1) bin.
pub struct CabacDecodeEngine<'a> {
    /// CABAC bytes for the current slice (post-emulation-prevention
    /// removed).
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

    /// Bins decoded so far. Diagnostic counter for trace cross-checks.
    bin_counts: u32,

    /// Optional per-bin decode trace, used for diff'ing the decode
    /// path against a reference bin sequence.
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

    /// Bit offset of the NEXT raw bit the engine will consume,
    /// measured from the start of the engine's input buffer
    /// (`byte_idx * 8 + bit_ptr`). For bypass-coded bins this is the
    /// position the next `decode_bypass()` call will read. For
    /// regular-coded bins the relationship to bin index is more
    /// complex (arithmetic coding may consume 0 or many raw bits per
    /// bin), so this is meaningful for **bypass-bin position capture**
    /// only.
    ///
    /// **Coordinate space**: the engine sees the slice's CABAC byte
    /// stream — i.e. `&nal.rbsp[cabac_byte_off..]` where
    /// `cabac_byte_off = cabac_data_byte_offset(slice_header.
    /// data_bit_offset)`. The returned offset is **engine-local**,
    /// NOT NAL-RBSP-absolute. Consumers that need the absolute
    /// position within `nal.rbsp` must add `cabac_byte_off * 8`; the
    /// streaming walker passes that base alongside `nal_idx` on each
    /// emit so downstream code (the bitstream-mod splicer) can
    /// compose the two.
    ///
    /// Used by the bitstream-mod stego path: the walker captures this
    /// offset alongside each stego-relevant bypass bin so a later
    /// post-encode pass can locate + flip that bit in the encoded
    /// stream.
    #[inline]
    pub fn next_rbsp_bit_offset(&self) -> u64 {
        (self.byte_idx as u64) * 8 + (self.bit_ptr as u64)
    }

    /// Variant that also carries a `ctx_idx` for trace logging.
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
