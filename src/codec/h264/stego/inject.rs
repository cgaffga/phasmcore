// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// CABAC bypass-bin stego primitives.
//
// Primitives that mirror the encoder's coefficient-sign / suffix-LSB /
// MVD bypass-bin emission order, so a caller can:
//   1. Enumerate the [`PositionKey`]s a residual block would produce
//      (used to build the cover position pool + per-domain capacity).
//   2. Extract the cover bits the decoder will see, given a decoded
//      coefficient / MVD vector.
//
// The original design also exposed in-place "apply
// overrides by flipping the cached coefficients" mutators (the
// pre-encode flip path driven by `GopDecisionCache`). Those retired
// with the pure-Rust encoder: the production OH264 path injects at
// the CABAC bypass-bin emit site (`wire_only`) instead, so these
// primitives now serve cover enumeration + extraction only.

use super::hook::{BinKind, EmbedDomain, PositionKey, SyntaxPath};



// ─── MvdSignBypass injection primitives ──────────────────────────
//
// Same pattern as the coefficient versions above, but for MVD signs.
// MVD has the additional wrinkle that an MVD value of 0 produces NO
// sign bypass bin (encoder skips the sign emission). The
// enumerator + applier handle that by filtering zero values out of
// the position list, mirroring the decoder.

/// Description of an MVD position in a macroblock. Used by
/// [`enumerate_mvd_sign_positions`] to reconstruct the
/// [`SyntaxPath::Mvd`] field set per partition.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MvdSlot {
    pub list: u8,
    pub partition: u8,
    pub axis: super::Axis,
    /// Signed MVD value the encoder will emit for this slot.
    pub value: i32,
}

/// Enumerate the [`PositionKey`]s for `MvdSignBypass` that the
/// encoder will produce for the given MVD slots, in the **same
/// order** the decoder emits them (slot iteration order). Slots
/// with value 0 produce NO key (no sign bypass bin emitted).
pub fn enumerate_mvd_sign_positions(
    slots: &[MvdSlot],
    frame_idx: u32,
    mb_addr: u32,
) -> Vec<PositionKey> {
    slots
        .iter()
        .filter(|s| s.value != 0)
        .map(|s| {
            let path = SyntaxPath::Mvd {
                list: s.list,
                partition: s.partition,
                axis: s.axis,
                kind: BinKind::Sign,
            };
            PositionKey::new(frame_idx, mb_addr, EmbedDomain::MvdSignBypass, path)
        })
        .collect()
}


/// Extract the `MvdSignBypass` cover bits the encoder would emit
/// for the given slots, in the **same order** as
/// [`enumerate_mvd_sign_positions`]. `0` for positive (or zero)
/// MVDs, `1` for negative.
pub fn extract_mvd_sign_bits(slots: &[MvdSlot]) -> Vec<u8> {
    slots
        .iter()
        .filter(|s| s.value != 0)
        .map(|s| if s.value < 0 { 1 } else { 0 })
        .collect()
}

// ─── Cross-domain orchestration ──────────────────────────────────

/// Per-domain summary of cover-side bits + positions for one GOP,
/// produced once the walker has visited every embeddable position.
/// The four `DomainBits` are concatenated by
/// [`super::combine_cover_4domain`] in canonical order
/// CS → CSL → MVDs → MVDsl into a single combined cover that drives
/// ONE [`crate::stego::stc`] plan (Scheme A);
/// [`super::split_plan_4domain`] reverses the concatenation to map
/// the plan bits back per domain.
#[derive(Default, Debug, Clone)]
pub struct DomainCover {
    pub coeff_sign_bypass: DomainBits,
    pub coeff_suffix_lsb: DomainBits,
    pub mvd_sign_bypass: DomainBits,
    pub mvd_suffix_lsb: DomainBits,
}

/// Cover bits + the positions that produced them, in emit order.
/// Indices align: `bits[i]` is the cover bit at `positions[i]`.
///
/// `magnitudes[i]` carries the absolute value of the coefficient
/// that emitted bit `i`, for CSB / CSL domains. MVD domains push `0`
/// here because their per-position
/// magnitude already lives in [`super::hook::MvdPositionMeta`].
/// Used by `compute_content_costs_yuv` so the J-UNIWARD-style cost
/// reflects the actual pixel delta a sign-flip introduces
/// (`δ = 2·|coeff|·Q-step·IDCT_basis`).
#[derive(Default, Debug, Clone)]
pub struct DomainBits {
    pub bits: Vec<u8>,
    pub positions: Vec<PositionKey>,
    pub magnitudes: Vec<u16>,
}

impl DomainBits {
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.bits.len(), self.positions.len());
        debug_assert_eq!(self.bits.len(), self.magnitudes.len());
        self.bits.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }
    /// Append one (bit, position) pair to this domain's cover.
    /// Used by Pass 1 scanners as they walk each block. Pushes
    /// `magnitude = 0`; for CSB / CSL use
    /// [`Self::push_with_magnitude`] instead.
    pub fn push(&mut self, bit: u8, pos: PositionKey) {
        self.push_with_magnitude(bit, pos, 0);
    }
    /// Same as `push` but records the coefficient magnitude
    /// alongside. Only meaningful for CoeffSignBypass /
    /// CoeffSuffixLsb positions; MVD callers stay on `push`.
    pub fn push_with_magnitude(&mut self, bit: u8, pos: PositionKey, magnitude: u16) {
        debug_assert!(bit <= 1);
        self.bits.push(bit);
        self.positions.push(pos);
        self.magnitudes.push(magnitude);
    }
    /// Reserve capacity for at least `n` more entries.
    /// Used by the walker once SPS is parsed to pre-size both inner
    /// `Vec`s up front. Eliminates the ~log2(n) doubling reallocs +
    /// memcopies on hot fixtures (1080p × 30f IMG_4138 IPPPP has
    /// ~514k CoeffSign positions; default Vec growth would copy
    /// ~24 MB cumulative).
    #[inline]
    pub fn reserve(&mut self, n: usize) {
        self.bits.reserve(n);
        self.positions.reserve(n);
        self.magnitudes.reserve(n);
    }
    /// Concatenate another DomainBits at the tail (same domain).
    /// Used to fold per-block contributions into the per-GOP cover.
    pub fn extend(&mut self, other: DomainBits) {
        self.bits.extend(other.bits);
        self.positions.extend(other.positions);
        self.magnitudes.extend(other.magnitudes);
    }
}

impl DomainCover {
    /// Total cover-bit count across all four domains.
    pub fn total_len(&self) -> usize {
        self.coeff_sign_bypass.len()
            + self.coeff_suffix_lsb.len()
            + self.mvd_sign_bypass.len()
            + self.mvd_suffix_lsb.len()
    }

    /// Concatenate another `DomainCover` at the tail (per-domain).
    /// Used by the whole-stream cover walk to fold per-slice covers
    /// into a single accumulated `DomainCover` across the stream.
    pub fn extend_from(&mut self, other: DomainCover) {
        self.coeff_sign_bypass.extend(other.coeff_sign_bypass);
        self.coeff_suffix_lsb.extend(other.coeff_suffix_lsb);
        self.mvd_sign_bypass.extend(other.mvd_sign_bypass);
        self.mvd_suffix_lsb.extend(other.mvd_suffix_lsb);
    }
}

// ─── CoeffSuffixLsb injection primitives ─────────────────────────
//
// Suffix LSB targets the LSB of the Exp-Golomb-0 suffix that the
// encoder emits when |coeff| ≥ 16 (abs_level_minus_1 ≥ 15, which
// saturates the TU prefix and routes through the suffix path).
//
// Eligibility threshold: |coeff| ≥ 16. Below 16, no suffix path,
// no LSB to target.
//
// Encoded LSB ↔ magnitude relation: encoded suffix LSB =
// `NOT (|coeff| & 1)` for both coefficients and MVDs (proof:
// `cabac-bypass-bin-stego.md` § 2.2). A flip changes |coeff| by ±1
// and never crosses the prefix length boundary (suffix value stays
// within `2^k_final`), so bitstream length is preserved.
//
// Direction selection: when |coeff| is exactly at the threshold
// (16 for coeffs, 9 for MVDs) the only valid direction is +1
// (going below would lose eligibility, breaking the position
// model). Otherwise prefer -1 (toward zero, lower distortion).

const COEFF_SUFFIX_LSB_THRESHOLD: u32 = 16;

/// Encoded suffix LSB bin value for a magnitude. Returns 0 when
/// |abs| has LSB 1, 1 when |abs| has LSB 0. Same formula for
/// CoeffSuffixLsb (k_init=0) and MvdSuffixLsb (k_init=3) because
/// the prefix offset is even for both eligible-range values.
#[inline]
fn suffix_lsb_bit_for_magnitude(abs: u32) -> u8 {
    ((abs & 1) ^ 1) as u8
}

/// Choose the new magnitude after flipping the suffix LSB. Always
/// changes by ±1; direction prefers -1 except at the threshold
/// where it must be +1 to preserve eligibility.
#[inline]
fn flipped_magnitude(abs: u32, threshold: u32) -> u32 {
    if abs == threshold { abs + 1 } else { abs - 1 }
}




/// Fused walk-and-push variant of the CoeffSign + CoeffSuffixLsb
/// position enumeration. Used by the decoder-side
/// walker's [`super::super::cabac::bin_decoder::positions::
/// PositionRecorder::on_residual_block`] hot path.
///
/// Walks `scan_coeffs[start_idx..=end_idx]` ONCE in reverse scan
/// order (matching the decoder emit order). For each significant
/// coefficient:
///   - Pushes the sign bit + CoeffSignBypass position into
///     `cover.coeff_sign_bypass`.
///   - If |coeff| ≥ 16 (`COEFF_SUFFIX_LSB_THRESHOLD`), also pushes
///     the suffix-LSB bit + CoeffSuffixLsb position into
///     `cover.coeff_suffix_lsb`.
///
/// Byte-identical to the legacy two-pass path that called
/// `enumerate_coeff_sign_positions` + `extract_coeff_sign_bits` +
/// `enumerate_coeff_suffix_lsb_positions` +
/// `extract_coeff_suffix_lsb_bits` separately. Confirmed by the 98
/// `cabac::bin_decoder` lib tests, all corpus round-trips, and the
/// `recorder_matches_encoder_position_logger_hook` parity gate.
///
/// **Why fused**: at 1080p × 30f IPPPP cover capture, the legacy
/// path allocates 4 intermediate `Vec`s per residual-block call (sig
/// indices + positions + sig indices again + bits, twice). With
/// ~16 blocks per MB × 241k MBs, that's ~3.86M small `malloc` hits.
/// The fused walker hits the allocator at most once per MB (the
/// per-MB `Vec::push` doubles which the pre-alloc in
/// `PositionRecorder::reserve_for_mb_count` further suppresses).
///
/// **Memory**: strictly reduces peak — no intermediate allocations
/// at all, and the destination `cover` Vecs are the same ones the
/// legacy path eventually pushed into.
pub fn record_residual_block_into_cover<F>(
    cover: &mut DomainCover,
    scan_coeffs: &[i32],
    start_idx: usize,
    end_idx: usize,
    frame_idx: u32,
    mb_addr: u32,
    mut path_for_coeff: F,
) where
    F: FnMut(u8, BinKind) -> SyntaxPath,
{
    // NEON fast path for the dominant 16-coefficient case
    // (4×4 luma/chroma scan, full range). Pre-computes a u16 bitmask
    // of non-zero positions in ~8 NEON instructions, then iterates
    // only the set bits MSB-first. Skips the per-position scalar
    // `if v == 0 { continue }` branch on 16 sparse positions.
    //
    // Byte-identical output to the scalar path — same MSB-first
    // (= reverse-scan) push order. Verified by the 98 walker lib
    // tests (incl. `recorder_matches_encoder_position_logger_hook`).
    //
    // Why only 16-element: it's the dominant call (4×4 transform).
    // 8×8 (64-element, High profile) falls through to scalar.
    // Partial ranges (Luma AC start=1, Chroma DC end=3, etc.) also
    // fall through. Extending to 64-element is a future v1.2+ item
    // if walker becomes hot again.
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        if end_idx >= start_idx
            && end_idx - start_idx == 15
            && start_idx + 16 <= scan_coeffs.len()
            && !simd_disabled_via_env()
        {
            // SAFETY: 16 contiguous i32s starting at start_idx are
            // in-bounds (checked above). NEON intrinsics here only
            // read the source slice; no aliasing concerns.
            let scan_block = &scan_coeffs[start_idx..start_idx + 16];
            let nz_mask: u16 = unsafe { nonzero_mask_neon_16(scan_block) };

            // Iterate set bits MSB-first to match scalar reverse-scan
            // emit order. `mask.leading_zeros()` is O(1) on aarch64
            // (CLZ instruction).
            let mut m: u32 = nz_mask as u32; // widen so we can use leading_zeros
            while m != 0 {
                let msb = 31 - m.leading_zeros() as usize; // position within u32; ≤ 15
                let i = start_idx + msb;
                let v = scan_coeffs[i];
                let ci = i as u8;
                let abs = v.unsigned_abs();

                let sign_path = with_sign_kind(path_for_coeff(ci, BinKind::Sign));
                let sign_key = PositionKey::new(
                    frame_idx, mb_addr, EmbedDomain::CoeffSignBypass, sign_path,
                );
                let sign_bit: u8 = if v < 0 { 1 } else { 0 };
                let abs_u16 = abs.min(u16::MAX as u32) as u16;
                cover.coeff_sign_bypass.push_with_magnitude(
                    sign_bit, sign_key, abs_u16,
                );

                if abs >= COEFF_SUFFIX_LSB_THRESHOLD {
                    let lsb_path = with_suffix_lsb_kind(
                        path_for_coeff(ci, BinKind::SuffixLsb),
                    );
                    let lsb_key = PositionKey::new(
                        frame_idx, mb_addr, EmbedDomain::CoeffSuffixLsb, lsb_path,
                    );
                    let lsb_bit = suffix_lsb_bit_for_magnitude(abs);
                    cover.coeff_suffix_lsb.push_with_magnitude(
                        lsb_bit, lsb_key, abs_u16,
                    );
                }
                m &= !(1u32 << msb);
            }
            return;
        }
    }

    // Scalar fallback (also handles 8×8, partial ranges, and
    // architectures without a NEON kernel).
    //
    // Reverse scan order matches the decoder's bypass-bin emission
    // order. The encoder also writes signs in reverse scan order.
    for i in (start_idx..=end_idx).rev() {
        let v = scan_coeffs[i];
        if v == 0 {
            continue;
        }
        let ci = i as u8;
        let abs = v.unsigned_abs();

        // CoeffSign: every non-zero coefficient contributes a sign
        // bypass bin.
        let sign_path = with_sign_kind(path_for_coeff(ci, BinKind::Sign));
        let sign_key = PositionKey::new(
            frame_idx, mb_addr, EmbedDomain::CoeffSignBypass, sign_path,
        );
        let sign_bit: u8 = if v < 0 { 1 } else { 0 };
        let abs_u16 = abs.min(u16::MAX as u32) as u16;
        cover.coeff_sign_bypass.push_with_magnitude(
            sign_bit, sign_key, abs_u16,
        );

        // CoeffSuffixLsb: only |coeff| ≥ 16 contributes.
        if abs >= COEFF_SUFFIX_LSB_THRESHOLD {
            let lsb_path = with_suffix_lsb_kind(
                path_for_coeff(ci, BinKind::SuffixLsb),
            );
            let lsb_key = PositionKey::new(
                frame_idx, mb_addr, EmbedDomain::CoeffSuffixLsb, lsb_path,
            );
            let lsb_bit = suffix_lsb_bit_for_magnitude(abs);
            cover.coeff_suffix_lsb.push_with_magnitude(
                lsb_bit, lsb_key, abs_u16,
            );
        }
    }
}

/// NEON kernel — compute a 16-bit mask where bit `i` is set
/// iff `scan[i] != 0`, for `scan.len() == 16`.
///
/// Implementation:
/// 1. Load 16 i32 values into 4 vectors (4 × `vld1q_s32`).
/// 2. `vceqzq_s32` → lane mask: 0xFFFFFFFF where zero, 0 where non-zero.
/// 3. `vmvnq_u32` → invert: 0xFFFFFFFF where non-zero, 0 where zero.
/// 4. AND each vector with a lane-specific power-of-2 stamp
///    (lanes 0-3 get bits 1-8, 4-7 get 16-128, 8-11 get 256-2048,
///     12-15 get 4096-32768).
/// 5. OR all 4 vectors together. Since each lane has a unique bit,
///    the OR forms the desired 16-bit mask spread across 4 u32 lanes.
/// 6. `vaddvq_u32` horizontal-sum across the 4 lanes — same as OR
///    here because the lanes have disjoint bit-fields.
///
/// Result is a u16 bitmask: bit `i` set iff `scan[i] != 0`.
///
/// **SAFETY**: caller must guarantee `scan.len() >= 16`. The function
/// reads exactly 16 i32 values.
/// PHASM_H264_DISABLE_SIMD=1 env var override (cached). The same
/// runtime A/B SIMD switch existed in the (retired) pure-Rust
/// encoder, but is implemented locally here so stego doesn't depend
/// on any encoder module — `stego::inject` is reachable from
/// non-h264-encoder builds too.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
#[inline]
fn simd_disabled_via_env() -> bool {
    use std::sync::OnceLock;
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("PHASM_H264_DISABLE_SIMD")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
#[inline]
unsafe fn nonzero_mask_neon_16(scan: &[i32]) -> u16 {
    use core::arch::aarch64::*;
    debug_assert!(scan.len() >= 16);
    let p = scan.as_ptr();
    let v0 = vld1q_s32(p);
    let v1 = vld1q_s32(p.add(4));
    let v2 = vld1q_s32(p.add(8));
    let v3 = vld1q_s32(p.add(12));
    let nz0 = vmvnq_u32(vceqzq_s32(v0));
    let nz1 = vmvnq_u32(vceqzq_s32(v1));
    let nz2 = vmvnq_u32(vceqzq_s32(v2));
    let nz3 = vmvnq_u32(vceqzq_s32(v3));
    // Lane-specific bit stamps. Lane 0..3 of v0 → bits 0..3, etc.
    let bits01: [u32; 4] = [1, 2, 4, 8];
    let bits23: [u32; 4] = [16, 32, 64, 128];
    let bits45: [u32; 4] = [256, 512, 1024, 2048];
    let bits67: [u32; 4] = [4096, 8192, 16384, 32768];
    let m0 = vandq_u32(nz0, vld1q_u32(bits01.as_ptr()));
    let m1 = vandq_u32(nz1, vld1q_u32(bits23.as_ptr()));
    let m2 = vandq_u32(nz2, vld1q_u32(bits45.as_ptr()));
    let m3 = vandq_u32(nz3, vld1q_u32(bits67.as_ptr()));
    let combined = vorrq_u32(vorrq_u32(m0, m1), vorrq_u32(m2, m3));
    // Each lane has unique bits set → sum-across == OR-across.
    vaddvq_u32(combined) as u16
}

// ─── MvdSuffixLsb injection primitives ───────────────────────────
//
// Same shape as the coefficient suffix-LSB primitives, but for MVD
// magnitudes ≥ 9 (the threshold at which the encoder emits an
// EGk-3 suffix).

pub(super) const MVD_SUFFIX_LSB_THRESHOLD: u32 = 9;

/// Direction-aware ±1 flip for an MvdSuffixLsb position, applying the
/// `flipped_magnitude(abs, MVD_SUFFIX_LSB_THRESHOLD)` direction
/// policy. Used by the wire-only path
/// (`InjectionHook::mvd_suffix_lsb_abs_override`) at the CABAC emit
/// site so the injected magnitude tracks the eligibility boundary.
///
/// Prefers `abs - 1` (smaller MV perturbation) when safe. At the
/// eligibility boundary (`abs == 9`), must return `abs + 1` to keep
/// the target magnitude in the UEG3 suffix-emitting range.
#[inline]
pub(super) fn mvd_flipped_magnitude(abs: u32) -> u32 {
    flipped_magnitude(abs, MVD_SUFFIX_LSB_THRESHOLD)
}

/// Enumerate `MvdSuffixLsb` positions for the given slots.
pub fn enumerate_mvd_suffix_lsb_positions(
    slots: &[MvdSlot],
    frame_idx: u32,
    mb_addr: u32,
) -> Vec<PositionKey> {
    slots
        .iter()
        .filter(|s| s.value.unsigned_abs() >= MVD_SUFFIX_LSB_THRESHOLD)
        .map(|s| {
            let path = SyntaxPath::Mvd {
                list: s.list,
                partition: s.partition,
                axis: s.axis,
                kind: BinKind::SuffixLsb,
            };
            PositionKey::new(frame_idx, mb_addr, EmbedDomain::MvdSuffixLsb, path)
        })
        .collect()
}


/// Extract `MvdSuffixLsb` cover bits.
pub fn extract_mvd_suffix_lsb_bits(slots: &[MvdSlot]) -> Vec<u8> {
    slots
        .iter()
        .filter(|s| s.value.unsigned_abs() >= MVD_SUFFIX_LSB_THRESHOLD)
        .map(|s| suffix_lsb_bit_for_magnitude(s.value.unsigned_abs()))
        .collect()
}

/// Force `kind = BinKind::SuffixLsb` on a SyntaxPath. Mirror of
/// [`with_sign_kind`] for the suffix-LSB domain.
fn with_suffix_lsb_kind(path: SyntaxPath) -> SyntaxPath {
    match path {
        SyntaxPath::Luma4x4 { block_idx, coeff_idx, .. } => SyntaxPath::Luma4x4 {
            block_idx, coeff_idx, kind: BinKind::SuffixLsb,
        },
        SyntaxPath::Luma8x8 { block_idx, coeff_idx, .. } => SyntaxPath::Luma8x8 {
            block_idx, coeff_idx, kind: BinKind::SuffixLsb,
        },
        SyntaxPath::ChromaAc { plane, block_idx, coeff_idx, .. } => SyntaxPath::ChromaAc {
            plane, block_idx, coeff_idx, kind: BinKind::SuffixLsb,
        },
        SyntaxPath::ChromaDc { plane, coeff_idx, .. } => SyntaxPath::ChromaDc {
            plane, coeff_idx, kind: BinKind::SuffixLsb,
        },
        SyntaxPath::LumaDcIntra16x16 { coeff_idx, .. } => SyntaxPath::LumaDcIntra16x16 {
            coeff_idx, kind: BinKind::SuffixLsb,
        },
        SyntaxPath::Mvd { list, partition, axis, .. } => SyntaxPath::Mvd {
            list, partition, axis, kind: BinKind::SuffixLsb,
        },
    }
}

/// Force `kind = BinKind::Sign` on a SyntaxPath built by a caller
/// closure, regardless of what they passed. Lets the
/// `path_for_coeff` closure focus on the structural fields
/// (block_idx, coeff_idx, plane, partition, axis) without having to
/// remember to set `kind` correctly each call.
fn with_sign_kind(path: SyntaxPath) -> SyntaxPath {
    match path {
        SyntaxPath::Luma4x4 { block_idx, coeff_idx, .. } => SyntaxPath::Luma4x4 {
            block_idx, coeff_idx, kind: BinKind::Sign,
        },
        SyntaxPath::Luma8x8 { block_idx, coeff_idx, .. } => SyntaxPath::Luma8x8 {
            block_idx, coeff_idx, kind: BinKind::Sign,
        },
        SyntaxPath::ChromaAc { plane, block_idx, coeff_idx, .. } => SyntaxPath::ChromaAc {
            plane, block_idx, coeff_idx, kind: BinKind::Sign,
        },
        SyntaxPath::ChromaDc { plane, coeff_idx, .. } => SyntaxPath::ChromaDc {
            plane, coeff_idx, kind: BinKind::Sign,
        },
        SyntaxPath::LumaDcIntra16x16 { coeff_idx, .. } => SyntaxPath::LumaDcIntra16x16 {
            coeff_idx, kind: BinKind::Sign,
        },
        SyntaxPath::Mvd { list, partition, axis, .. } => SyntaxPath::Mvd {
            list, partition, axis, kind: BinKind::Sign,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luma4x4_path(coeff_idx: u8) -> SyntaxPath {
        SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx, kind: BinKind::Sign }
    }


    #[test]
    fn mvd_enumerate_skips_zero_values() {
        let slots = vec![
            MvdSlot { list: 0, partition: 0, axis: super::super::Axis::X, value: 5 },
            MvdSlot { list: 0, partition: 0, axis: super::super::Axis::Y, value: 0 },
            MvdSlot { list: 0, partition: 1, axis: super::super::Axis::X, value: -3 },
            MvdSlot { list: 0, partition: 1, axis: super::super::Axis::Y, value: 0 },
        ];
        let positions = enumerate_mvd_sign_positions(&slots, 5, 100);
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].frame_idx(), 5);
        assert_eq!(positions[0].mb_addr(), 100);
        for k in &positions {
            assert_eq!(k.domain(), EmbedDomain::MvdSignBypass);
        }
    }

    #[test]
    fn mvd_extract_skips_zero_values() {
        let slots = vec![
            MvdSlot { list: 0, partition: 0, axis: super::super::Axis::X, value: 5 },
            MvdSlot { list: 0, partition: 0, axis: super::super::Axis::Y, value: 0 },
            MvdSlot { list: 0, partition: 1, axis: super::super::Axis::X, value: -3 },
        ];
        let bits = extract_mvd_sign_bits(&slots);
        assert_eq!(bits, vec![0, 1]);
    }


    #[test]
    fn mvd_suffix_lsb_below_threshold_not_eligible() {
        let slots = vec![
            MvdSlot { list: 0, partition: 0, axis: super::super::Axis::X, value: 8 },
        ];
        let positions = enumerate_mvd_suffix_lsb_positions(&slots, 0, 0);
        assert!(positions.is_empty());
    }

    #[test]
    fn mvd_suffix_lsb_threshold_eligible() {
        let slots = vec![
            MvdSlot { list: 0, partition: 0, axis: super::super::Axis::X, value: 9 },
            MvdSlot { list: 0, partition: 0, axis: super::super::Axis::Y, value: -10 },
        ];
        let positions = enumerate_mvd_suffix_lsb_positions(&slots, 0, 0);
        assert_eq!(positions.len(), 2);
        let bits = extract_mvd_suffix_lsb_bits(&slots);
        // |9| → NOT 1 = 0. |10| → NOT 0 = 1.
        assert_eq!(bits, vec![0, 1]);
    }
}
