// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Spec-literal CABAC reference encoder. Implements H.264 § 9.3.4
// EXACTLY as written: PutBit bit-by-bit (Figure 9-11), RenormE
// (Figure 9-8), EncodeTerminate + EncodeFlush (Figures 9-11 / 9-12).
//
// Purpose: diff against our byte-buffered CabacEngine for the same
// bin sequence, localize where the two diverge.
//
// Run: cargo run --release --example h264_cabac_spec_ref

use phasm_core::codec::h264::cabac::context::CabacContext;
use phasm_core::codec::h264::cabac::engine::CabacEngine;
use phasm_core::codec::h264::cabac::tables::RANGE_TAB_LPS;

/// Spec-literal CABAC encoder. Uses BIT-level PutBit per spec
/// § 9.3.4.5 Figure 9-11 — no byte buffering. Output is a Vec<u8>
/// with bits packed MSB-first (same layout as our engine produces).
struct SpecEncoder {
    cod_i_range: u32,
    cod_i_low: u32,
    first_bit_flag: bool,
    bits_outstanding: u32,
    /// Output byte stream. Bits are packed MSB-first into this vec.
    bit_out: BitWriter,
}

struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    cur_bits: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self { bytes: Vec::new(), cur: 0, cur_bits: 0 }
    }

    fn write_bit(&mut self, bit: u32) {
        self.cur = (self.cur << 1) | ((bit & 1) as u8);
        self.cur_bits += 1;
        if self.cur_bits == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.cur_bits = 0;
        }
    }

    /// Byte-align by padding with zeros. RBSP stop bit is emitted
    /// inside EncodeFlush (it's the LSB of the trailing WriteBits
    /// pattern `((low>>7)&3)|1`), NOT added here.
    fn finalize(mut self) -> Vec<u8> {
        while self.cur_bits != 0 {
            self.write_bit(0);
        }
        self.bytes
    }
}

impl SpecEncoder {
    fn new() -> Self {
        Self {
            cod_i_range: 510,
            cod_i_low: 0,
            first_bit_flag: true,
            bits_outstanding: 0,
            bit_out: BitWriter::new(),
        }
    }

    /// § 9.3.4.2 EncodeDecision for regular (context-coded) bins.
    fn encode_decision(&mut self, bin: u8, ctx: &mut CabacContext) {
        let p_state = ctx.p_state_idx() as usize;
        let q_idx = ((self.cod_i_range >> 6) & 3) as usize;
        let range_lps = RANGE_TAB_LPS[p_state][q_idx] as u32;
        self.cod_i_range -= range_lps;

        if bin != ctx.val_mps() {
            self.cod_i_low = self.cod_i_low.wrapping_add(self.cod_i_range);
            self.cod_i_range = range_lps;
            ctx.update_lps();
        } else {
            ctx.update_mps();
        }

        self.renorm_e();
    }

    /// § 9.3.4.4 EncodeBypass.
    fn encode_bypass(&mut self, bin: u8) {
        self.cod_i_low <<= 1;
        if bin != 0 {
            self.cod_i_low = self.cod_i_low.wrapping_add(self.cod_i_range);
        }
        // After one bit, test for "double carry" per spec Figure 9-10:
        if self.cod_i_low >= 0x400 {
            self.put_bit(1);
            self.cod_i_low -= 0x400;
        } else if self.cod_i_low < 0x200 {
            self.put_bit(0);
        } else {
            self.cod_i_low -= 0x200;
            self.bits_outstanding += 1;
        }
    }

    /// § 9.3.4.5 EncodeTerminate. When bin=1, invokes EncodeFlush
    /// (Figure 9-12) which also emits the RBSP stop bit as part of
    /// the trailing WriteBits pattern.
    fn encode_terminate(&mut self, bin: u8) {
        self.cod_i_range -= 2;
        if bin != 0 {
            self.cod_i_low = self.cod_i_low.wrapping_add(self.cod_i_range);
            // EncodeFlush:
            self.cod_i_range = 2;
            self.renorm_e();
            self.put_bit((self.cod_i_low >> 9) & 1);
            // WriteBits(((cod_i_low >> 7) & 3) | 1, 2) — MSB-first.
            let val = ((self.cod_i_low >> 7) & 3) | 1;
            self.bit_out.write_bit((val >> 1) & 1);
            self.bit_out.write_bit(val & 1);
        } else {
            self.renorm_e();
        }
    }

    /// § 9.3.4.3 Figure 9-8 RenormE.
    fn renorm_e(&mut self) {
        while self.cod_i_range < 0x100 {
            if self.cod_i_low < 0x100 {
                self.put_bit(0);
            } else if self.cod_i_low >= 0x200 {
                self.cod_i_low -= 0x200;
                self.put_bit(1);
            } else {
                self.cod_i_low -= 0x100;
                self.bits_outstanding += 1;
            }
            self.cod_i_range <<= 1;
            self.cod_i_low <<= 1;
        }
    }

    /// § 9.3.4.5 Figure 9-11 PutBit.
    fn put_bit(&mut self, b: u32) {
        if self.first_bit_flag {
            self.first_bit_flag = false;
        } else {
            self.bit_out.write_bit(b);
        }
        let inv = 1 - b;
        while self.bits_outstanding > 0 {
            self.bit_out.write_bit(inv);
            self.bits_outstanding -= 1;
        }
    }

    fn finish(self) -> Vec<u8> {
        self.bit_out.finalize()
    }
}

/// The 11-bin sequence we care about for intra-in-P MB (0,0) in P-slice:
/// - decision bin at ctx=11 (mb_skip_flag=0), mps-state (22, 0)
/// - decision bin at ctx=14 (mb_type_p prefix bin 0 = 1)
/// - decision bin at ctx=17 (mb_type_p prefix bin 1 = 1)
/// - terminate bin 0 (non-final terminate for end-of-prefix)
/// - decision bin at ctx=18 (mb_type_i suffix bin 0 = 1)
/// - decision bin at ctx=19 (mb_type_i suffix bin 1 = 1)
/// - decision bin at ctx=19 (mb_type_i suffix bin 2 = 0)
/// - decision bin at ctx=20 (mb_type_i suffix bin 3 = 1)
/// - decision bin at ctx=20 (mb_type_i suffix bin 4 = 0)
/// - decision bin at ctx=64 (intra_chroma_pred_mode bin 0 = 0)
/// - decision bin at ctx=60 (mb_qp_delta bin 0 = 0)
fn run_diverge_sequence() -> (Vec<u8>, Vec<u8>) {
    // Contexts initialized to match trace pre-states. From the trace
    // table, the pre-states at bin-encode time are:
    //   ctx=11 → p_state=22, val_mps=0
    //   ctx=14 → p_state=53, val_mps=0
    //   ctx=17 → p_state=0,  val_mps=0
    //   ctx=18 → p_state=3,  val_mps=0
    //   ctx=19 → p_state=14, val_mps=0 (first entry)
    //         → p_state=11, val_mps=0 (second entry, after update_mps)
    //   ctx=20 → p_state=0,  val_mps=0 (first)
    //         → p_state=0,  val_mps=1 (second, after update_lps flips val_mps)
    //   ctx=64 → p_state=6,  val_mps=1
    //   ctx=60 → p_state=22, val_mps=0
    // These come from the P-slice CABAC init table at QP~=26.
    // For the test, we construct fresh contexts at these pre-states
    // and run the bins through both encoders.

    // spec-literal encoder
    let mut spec = SpecEncoder::new();
    let mut ctx11 = CabacContext::new(22, 0);
    let mut ctx14 = CabacContext::new(53, 0);
    let mut ctx17 = CabacContext::new(0, 0);
    let mut ctx18 = CabacContext::new(3, 0);
    let mut ctx19 = CabacContext::new(14, 0);
    let mut ctx20 = CabacContext::new(0, 0);
    let mut ctx64 = CabacContext::new(6, 1);
    let mut ctx60 = CabacContext::new(22, 0);

    spec.encode_decision(0, &mut ctx11);
    spec.encode_decision(1, &mut ctx14);
    spec.encode_decision(1, &mut ctx17);
    spec.encode_terminate(0);
    spec.encode_decision(1, &mut ctx18);
    spec.encode_decision(1, &mut ctx19);
    spec.encode_decision(0, &mut ctx19);
    spec.encode_decision(1, &mut ctx20);
    spec.encode_decision(0, &mut ctx20);
    spec.encode_decision(0, &mut ctx64);
    spec.encode_decision(0, &mut ctx60);
    spec.encode_terminate(1);
    let spec_bytes = spec.finish();

    // byte-buffered engine
    let mut eng = CabacEngine::new();
    let mut ctx11 = CabacContext::new(22, 0);
    let mut ctx14 = CabacContext::new(53, 0);
    let mut ctx17 = CabacContext::new(0, 0);
    let mut ctx18 = CabacContext::new(3, 0);
    let mut ctx19 = CabacContext::new(14, 0);
    let mut ctx20 = CabacContext::new(0, 0);
    let mut ctx64 = CabacContext::new(6, 1);
    let mut ctx60 = CabacContext::new(22, 0);

    eng.encode_decision(0, &mut ctx11);
    eng.encode_decision(1, &mut ctx14);
    eng.encode_decision(1, &mut ctx17);
    eng.encode_terminate(0);
    eng.encode_decision(1, &mut ctx18);
    eng.encode_decision(1, &mut ctx19);
    eng.encode_decision(0, &mut ctx19);
    eng.encode_decision(1, &mut ctx20);
    eng.encode_decision(0, &mut ctx20);
    eng.encode_decision(0, &mut ctx64);
    eng.encode_decision(0, &mut ctx60);
    eng.encode_terminate(1);
    let eng_bytes = eng.finish();

    (spec_bytes, eng_bytes)
}

/// Generate a long pseudo-random bin sequence and encode it through
/// both encoders, diffing byte-by-byte. Uses a seed-controlled
/// deterministic RNG so the test is reproducible.
fn run_long_random_sequence(seed: u32, n_bins: usize) -> (Vec<u8>, Vec<u8>) {
    fn next(state: &mut u32) -> u32 {
        // xorshift32
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state
    }

    let mut rng = seed.max(1);

    // Encode with the spec-literal encoder.
    let mut spec = SpecEncoder::new();
    let mut ctxs_spec: Vec<CabacContext> = (0..64)
        .map(|i| CabacContext::new((next(&mut rng) % 63) as u8, (i & 1) as u8))
        .collect();
    let mut rng_spec = seed.max(1);
    let mut _rng_spec_ctx = seed.max(1);

    // Reset rng for a fresh bin-gen walk, to keep the two walks in
    // lockstep (same bin sequence fed to both encoders).
    rng_spec = (seed ^ 0xdeadbeef).max(1);
    for i in 0..n_bins {
        let bin = (next(&mut rng_spec) & 1) as u8;
        // Mix in bypass bins ~30% of the time; terminate(0) ~2%.
        let op = next(&mut rng_spec) % 100;
        if op < 30 {
            spec.encode_bypass(bin);
        } else if op < 32 && i < n_bins - 5 {
            spec.encode_terminate(0);
        } else {
            let ctx_idx = (next(&mut rng_spec) as usize) % 64;
            spec.encode_decision(bin, &mut ctxs_spec[ctx_idx]);
        }
    }
    spec.encode_terminate(1);
    let spec_bytes = spec.finish();

    // Encode with the byte-buffered engine — same seeds, same state.
    let mut eng = CabacEngine::new();
    let mut ctxs_eng: Vec<CabacContext> = {
        let mut r = seed.max(1);
        (0..64)
            .map(|i| CabacContext::new((next(&mut r) % 63) as u8, (i & 1) as u8))
            .collect()
    };
    let mut rng_eng = (seed ^ 0xdeadbeef).max(1);
    for i in 0..n_bins {
        let bin = (next(&mut rng_eng) & 1) as u8;
        let op = next(&mut rng_eng) % 100;
        if op < 30 {
            eng.encode_bypass(bin);
        } else if op < 32 && i < n_bins - 5 {
            eng.encode_terminate(0);
        } else {
            let ctx_idx = (next(&mut rng_eng) as usize) % 64;
            eng.encode_decision(bin, &mut ctxs_eng[ctx_idx]);
        }
    }
    eng.encode_terminate(1);
    let eng_bytes = eng.finish();

    let _ = _rng_spec_ctx;
    (spec_bytes, eng_bytes)
}

fn main() {
    // Test 1: short 11-bin divergence sequence.
    let (spec_bytes, eng_bytes) = run_diverge_sequence();
    println!("=== Short (11-bin) sequence ===");
    println!("spec-ref encoder bytes: {:02x?}", spec_bytes);
    println!("byte-buf  encoder bytes: {:02x?}", eng_bytes);
    if spec_bytes == eng_bytes {
        println!("MATCH");
    } else {
        let first_diff = spec_bytes.iter().zip(eng_bytes.iter()).take_while(|(a, b)| a == b).count();
        println!("DIFFER at byte {} (lens {} vs {})", first_diff, spec_bytes.len(), eng_bytes.len());
    }

    // Test 2: long pseudo-random sequences with several seeds.
    println!("\n=== Long (1500-bin) pseudo-random sequences ===");
    for seed in [1u32, 42, 0xdeadu32, 0xabadcafe, 0xfeedface] {
        let (spec, eng) = run_long_random_sequence(seed, 1500);
        if spec == eng {
            println!("seed 0x{:08x}: MATCH ({} bytes)", seed, spec.len());
        } else {
            let first_diff = spec.iter().zip(eng.iter()).take_while(|(a, b)| a == b).count();
            println!(
                "seed 0x{:08x}: DIFFER at byte {} — spec len {}, eng len {}",
                seed, first_diff, spec.len(), eng.len()
            );
            let a_str: String = spec.iter().skip(first_diff.saturating_sub(2)).take(8).map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
            let b_str: String = eng.iter().skip(first_diff.saturating_sub(2)).take(8).map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
            println!("  spec: ...{}...", a_str);
            println!("  eng:  ...{}...", b_str);
        }
    }
}
