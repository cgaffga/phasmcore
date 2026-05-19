// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// CoeffSignBypass injection helpers (Phase 6D.3).
//
// Pre-encode primitives that mirror the encoder's coefficient-sign
// bypass-bin emission order, so a caller can:
//   1. Enumerate the [`PositionKey`]s a residual block would produce.
//   2. Apply a [`BitInjector`]'s overrides by flipping coefficient
//      signs in-place (zero magnitude impact, zero rate impact —
//      bypass bin → bypass bin).
//   3. Extract the sign bits the decoder will see, given a decoded
//      coefficient vector.
//
// This is the **pre-encode** approach to 6D.3: stego flips happen on
// the cached quantized coefficients before the encoder runs Pass 3.
// Final wiring into the encoder hot path (an in-place hook at the
// bypass-bin emit site) lands in 6D.7 alongside the
// `GopDecisionCache` lifecycle in the three-pass driver. The two
// approaches produce byte-identical output because both flip the
// same coefficients before entropy coding.

use super::hook::{BinKind, BitInjector, EmbedDomain, PositionKey, SyntaxPath};

/// Enumerate the [`PositionKey`]s for `CoeffSignBypass` that a
/// residual block at `(frame_idx, mb_addr)` with `scan_coeffs`
/// would produce, in the **same order** the decoder emits them
/// (reverse scan order over significant positions, from the highest
/// scan index down to `start_idx`).
///
/// `path_for_coeff(coeff_idx)` builds a [`SyntaxPath`] for the given
/// coefficient. The caller knows which block type (Luma4x4,
/// Luma8x8, ChromaAc, ChromaDc, LumaDcIntra16x16) is being processed
/// and supplies the appropriate constructor. The constructor's
/// `kind` field will be set to `BinKind::Sign` regardless of what
/// the constructor returns.
pub fn enumerate_coeff_sign_positions(
    scan_coeffs: &[i32],
    start_idx: usize,
    end_idx: usize,
    frame_idx: u32,
    mb_addr: u32,
    mut path_for_coeff: impl FnMut(u8) -> SyntaxPath,
) -> Vec<PositionKey> {
    let mut sig: Vec<usize> = (start_idx..=end_idx)
        .filter(|&i| scan_coeffs[i] != 0)
        .collect();
    // Encoder + decoder both emit signs in reverse scan order.
    sig.reverse();
    sig.into_iter()
        .map(|i| {
            let path = with_sign_kind(path_for_coeff(i as u8));
            PositionKey::new(frame_idx, mb_addr, EmbedDomain::CoeffSignBypass, path)
        })
        .collect()
}

/// Apply [`BitInjector`] overrides for `CoeffSignBypass` to a
/// residual block's `scan_coeffs` in place. For each significant
/// coefficient (in reverse scan order, mirroring the decoder),
/// query the injector. If it returns `Some(bit)`, ensure the
/// coefficient's encoded sign matches `bit` (0 = positive, 1 =
/// negative — same convention as the bypass bin written to the
/// bitstream). Magnitudes are never changed.
///
/// Returns the number of positions where the coefficient sign was
/// flipped to satisfy an override (the "modification count").
/// Diagnostic; production callers can use this to track STC
/// distortion.
pub fn apply_coeff_sign_overrides(
    scan_coeffs: &mut [i32],
    start_idx: usize,
    end_idx: usize,
    frame_idx: u32,
    mb_addr: u32,
    mut path_for_coeff: impl FnMut(u8) -> SyntaxPath,
    injector: &mut dyn BitInjector,
) -> usize {
    let mut count = 0usize;
    let mut sig: Vec<usize> = (start_idx..=end_idx)
        .filter(|&i| scan_coeffs[i] != 0)
        .collect();
    sig.reverse();
    for i in sig {
        let path = with_sign_kind(path_for_coeff(i as u8));
        let key = PositionKey::new(frame_idx, mb_addr, EmbedDomain::CoeffSignBypass, path);
        if let Some(bit) = injector.override_bit(key) {
            let want_negative = bit == 1;
            let is_negative = scan_coeffs[i] < 0;
            if want_negative != is_negative {
                scan_coeffs[i] = -scan_coeffs[i];
                count += 1;
            }
        }
    }
    count
}

/// Extract the `CoeffSignBypass` cover bits a residual block would
/// emit for `scan_coeffs`, in the **same order** as
/// [`enumerate_coeff_sign_positions`]. Each bit is `0` for positive
/// coefficients and `1` for negative (matching the bypass-bin
/// convention).
///
/// Used by:
/// - Pass 1 (capacity sizing) to extract cover bits per domain.
/// - Tests to verify decoder ↔ extractor parity on the same scan.
pub fn extract_coeff_sign_bits(
    scan_coeffs: &[i32],
    start_idx: usize,
    end_idx: usize,
) -> Vec<u8> {
    let mut sig: Vec<usize> = (start_idx..=end_idx)
        .filter(|&i| scan_coeffs[i] != 0)
        .collect();
    sig.reverse();
    sig.into_iter()
        .map(|i| if scan_coeffs[i] < 0 { 1 } else { 0 })
        .collect()
}

// ─── MvdSignBypass injection primitives ──────────────────────────
//
// Same pattern as the coefficient versions above, but for MVD signs.
// MVD has the additional wrinkle that an MVD value of 0 produces NO
// sign bypass bin (encoder skips the sign emission). The
// enumerator + applier handle that by filtering zero values out of
// the position list, mirroring the decoder.

/// Description of an MVD position in a macroblock. Used by
/// [`enumerate_mvd_sign_positions`] / [`apply_mvd_sign_overrides`]
/// to reconstruct the [`SyntaxPath::Mvd`] field set per partition.
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

/// Apply [`BitInjector`] overrides for `MvdSignBypass` to a list of
/// MVD slots in place. For each non-zero slot, query the injector;
/// if it returns `Some(bit)`, force `value`'s sign to match (`bit=0`
/// → positive, `bit=1` → negative). Magnitudes are never changed.
/// Returns the number of sign flips that occurred.
pub fn apply_mvd_sign_overrides(
    slots: &mut [MvdSlot],
    frame_idx: u32,
    mb_addr: u32,
    injector: &mut dyn BitInjector,
) -> usize {
    let mut count = 0usize;
    for s in slots.iter_mut() {
        if s.value == 0 {
            continue;
        }
        let path = SyntaxPath::Mvd {
            list: s.list,
            partition: s.partition,
            axis: s.axis,
            kind: BinKind::Sign,
        };
        let key = PositionKey::new(frame_idx, mb_addr, EmbedDomain::MvdSignBypass, path);
        if let Some(bit) = injector.override_bit(key) {
            let want_negative = bit == 1;
            let is_negative = s.value < 0;
            if want_negative != is_negative {
                s.value = -s.value;
                count += 1;
            }
        }
    }
    count
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

/// Per-domain summary of cover-side bits + positions, ready to feed
/// into [`crate::stego::stc`] embedding. Returned by Pass 1's
/// scanner once it has visited every embeddable position in a GOP.
///
/// The four-domain split keeps each domain's STC plan independent
/// (separate ChaCha20 keys, separate w-selection per domain) so an
/// attacker breaking one domain's syndrome can't reach any other.
#[derive(Default, Debug, Clone)]
pub struct DomainCover {
    pub coeff_sign_bypass: DomainBits,
    pub coeff_suffix_lsb: DomainBits,
    pub mvd_sign_bypass: DomainBits,
    pub mvd_suffix_lsb: DomainBits,
}

/// Cover bits + the positions that produced them, in emit order.
/// Indices align: `bits[i]` is the cover bit at `positions[i]`.
#[derive(Default, Debug, Clone)]
pub struct DomainBits {
    pub bits: Vec<u8>,
    pub positions: Vec<PositionKey>,
}

impl DomainBits {
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.bits.len(), self.positions.len());
        self.bits.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }
    /// Append one (bit, position) pair to this domain's cover.
    /// Used by Pass 1 scanners as they walk each block.
    pub fn push(&mut self, bit: u8, pos: PositionKey) {
        debug_assert!(bit <= 1);
        self.bits.push(bit);
        self.positions.push(pos);
    }
    /// #516.1 perf — reserve capacity for at least `n` more entries.
    /// Used by the walker once SPS is parsed to pre-size both inner
    /// `Vec`s up front. Eliminates the ~log2(n) doubling reallocs +
    /// memcopies on hot fixtures (1080p × 30f IMG_4138 IPPPP has
    /// ~514k CoeffSign positions; default Vec growth would copy
    /// ~24 MB cumulative).
    #[inline]
    pub fn reserve(&mut self, n: usize) {
        self.bits.reserve(n);
        self.positions.reserve(n);
    }
    /// Concatenate another DomainBits at the tail (same domain).
    /// Used to fold per-block contributions into the per-GOP cover.
    pub fn extend(&mut self, other: DomainBits) {
        self.bits.extend(other.bits);
        self.positions.extend(other.positions);
    }
    /// Truncate to the given length (saved earlier via `len()`).
    /// Used by Phase 6F.2's MVD-rollback path: when the encoder
    /// ends up emitting an MB as P_SKIP / intra-in-P after the
    /// MVD hook already logged positions, those phantom positions
    /// must be retracted to keep the cover in sync with the actual
    /// bitstream. See `PositionLoggerHook::rollback_mvd_for_mb`.
    pub fn truncate(&mut self, new_len: usize) {
        self.bits.truncate(new_len);
        self.positions.truncate(new_len);
    }
}

impl DomainCover {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total cover-bit count across all four domains.
    pub fn total_len(&self) -> usize {
        self.coeff_sign_bypass.len()
            + self.coeff_suffix_lsb.len()
            + self.mvd_sign_bypass.len()
            + self.mvd_suffix_lsb.len()
    }

    /// Per-domain capacity vector for STC sizing. Mirror of
    /// [`super::GopCapacity`].
    pub fn capacity(&self) -> super::GopCapacity {
        super::GopCapacity {
            coeff_sign_bypass: self.coeff_sign_bypass.len(),
            coeff_suffix_lsb: self.coeff_suffix_lsb.len(),
            mvd_sign_bypass: self.mvd_sign_bypass.len(),
            mvd_suffix_lsb: self.mvd_suffix_lsb.len(),
        }
    }

    /// Mutable accessor for the matching domain. Used by Pass 1
    /// per-block scanners to dispatch their per-domain emissions.
    pub fn for_domain_mut(&mut self, domain: EmbedDomain) -> &mut DomainBits {
        match domain {
            EmbedDomain::CoeffSignBypass => &mut self.coeff_sign_bypass,
            EmbedDomain::CoeffSuffixLsb => &mut self.coeff_suffix_lsb,
            EmbedDomain::MvdSignBypass => &mut self.mvd_sign_bypass,
            EmbedDomain::MvdSuffixLsb => &mut self.mvd_suffix_lsb,
        }
    }

    /// Concatenate another `DomainCover` at the tail (per-domain).
    /// Used by the streaming walker's wrapper to fold per-GOP
    /// covers into a single accumulated cover for backwards-compat
    /// with the legacy whole-stream API.
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

/// Enumerate the [`PositionKey`]s for `CoeffSuffixLsb` that the
/// encoder will produce for a residual block, in the same order
/// the decoder emits them (reverse scan order over significant
/// positions where |coeff| ≥ 16).
pub fn enumerate_coeff_suffix_lsb_positions(
    scan_coeffs: &[i32],
    start_idx: usize,
    end_idx: usize,
    frame_idx: u32,
    mb_addr: u32,
    mut path_for_coeff: impl FnMut(u8) -> SyntaxPath,
) -> Vec<PositionKey> {
    let mut sig: Vec<usize> = (start_idx..=end_idx)
        .filter(|&i| scan_coeffs[i].unsigned_abs() >= COEFF_SUFFIX_LSB_THRESHOLD)
        .collect();
    sig.reverse();
    sig.into_iter()
        .map(|i| {
            let path = with_suffix_lsb_kind(path_for_coeff(i as u8));
            PositionKey::new(frame_idx, mb_addr, EmbedDomain::CoeffSuffixLsb, path)
        })
        .collect()
}

/// Apply [`BitInjector`] overrides for `CoeffSuffixLsb` to a
/// residual block's `scan_coeffs` in place. Each override flips
/// |coeff| by ±1; direction is chosen to keep |coeff| above the
/// eligibility threshold (16). Sign is preserved.
/// Returns the modification count.
pub fn apply_coeff_suffix_lsb_overrides(
    scan_coeffs: &mut [i32],
    start_idx: usize,
    end_idx: usize,
    frame_idx: u32,
    mb_addr: u32,
    mut path_for_coeff: impl FnMut(u8) -> SyntaxPath,
    injector: &mut dyn BitInjector,
) -> usize {
    let mut count = 0usize;
    let mut sig: Vec<usize> = (start_idx..=end_idx)
        .filter(|&i| scan_coeffs[i].unsigned_abs() >= COEFF_SUFFIX_LSB_THRESHOLD)
        .collect();
    sig.reverse();
    for i in sig {
        let path = with_suffix_lsb_kind(path_for_coeff(i as u8));
        let key = PositionKey::new(frame_idx, mb_addr, EmbedDomain::CoeffSuffixLsb, path);
        if let Some(target_bit) = injector.override_bit(key) {
            let abs = scan_coeffs[i].unsigned_abs();
            let cover_bit = suffix_lsb_bit_for_magnitude(abs);
            if target_bit != cover_bit {
                let new_abs = flipped_magnitude(abs, COEFF_SUFFIX_LSB_THRESHOLD);
                scan_coeffs[i] = if scan_coeffs[i] < 0 {
                    -(new_abs as i32)
                } else {
                    new_abs as i32
                };
                count += 1;
            }
        }
    }
    count
}

/// Extract the `CoeffSuffixLsb` cover bits for a residual block.
pub fn extract_coeff_suffix_lsb_bits(
    scan_coeffs: &[i32],
    start_idx: usize,
    end_idx: usize,
) -> Vec<u8> {
    let mut sig: Vec<usize> = (start_idx..=end_idx)
        .filter(|&i| scan_coeffs[i].unsigned_abs() >= COEFF_SUFFIX_LSB_THRESHOLD)
        .collect();
    sig.reverse();
    sig.into_iter()
        .map(|i| suffix_lsb_bit_for_magnitude(scan_coeffs[i].unsigned_abs()))
        .collect()
}

/// #516.1.b perf — fused walk-and-push variant of the CoeffSign +
/// CoeffSuffixLsb position enumeration. Used by the decoder-side
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
    // #516.C — NEON fast path for the dominant 16-coefficient case
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
                cover.coeff_sign_bypass.push(sign_bit, sign_key);

                if abs >= COEFF_SUFFIX_LSB_THRESHOLD {
                    let lsb_path = with_suffix_lsb_kind(
                        path_for_coeff(ci, BinKind::SuffixLsb),
                    );
                    let lsb_key = PositionKey::new(
                        frame_idx, mb_addr, EmbedDomain::CoeffSuffixLsb, lsb_path,
                    );
                    let lsb_bit = suffix_lsb_bit_for_magnitude(abs);
                    cover.coeff_suffix_lsb.push(lsb_bit, lsb_key);
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
        cover.coeff_sign_bypass.push(sign_bit, sign_key);

        // CoeffSuffixLsb: only |coeff| ≥ 16 contributes.
        if abs >= COEFF_SUFFIX_LSB_THRESHOLD {
            let lsb_path = with_suffix_lsb_kind(
                path_for_coeff(ci, BinKind::SuffixLsb),
            );
            let lsb_key = PositionKey::new(
                frame_idx, mb_addr, EmbedDomain::CoeffSuffixLsb, lsb_path,
            );
            let lsb_bit = suffix_lsb_bit_for_magnitude(abs);
            cover.coeff_suffix_lsb.push(lsb_bit, lsb_key);
        }
    }
}

/// #516.C NEON kernel — compute a 16-bit mask where bit `i` is set
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
/// PHASM_H264_DISABLE_SIMD=1 env var override (cached). Mirrors the
/// runtime A/B switch in `encoder::simd::simd_enabled` but
/// implemented locally so stego doesn't depend on the encoder
/// module — `stego::inject` is reachable from non-h264-encoder
/// builds too.
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

/// Direction-aware ±1 flip for an MvdSuffixLsb position, mirror of
/// the local `flipped_magnitude(abs, MVD_SUFFIX_LSB_THRESHOLD)` call
/// in [`apply_mvd_suffix_lsb_overrides`]. Exposed for the wire-only
/// path (#540.1, `InjectionHook::mvd_suffix_lsb_abs_override`) so
/// both mutation and wire-only code share one direction policy.
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

/// Apply [`BitInjector`] overrides for `MvdSuffixLsb` to a slot
/// list. Direction-aware ±1 flip preserving eligibility.
pub fn apply_mvd_suffix_lsb_overrides(
    slots: &mut [MvdSlot],
    frame_idx: u32,
    mb_addr: u32,
    injector: &mut dyn BitInjector,
) -> usize {
    let mut count = 0usize;
    for s in slots.iter_mut() {
        let abs = s.value.unsigned_abs();
        if abs < MVD_SUFFIX_LSB_THRESHOLD {
            continue;
        }
        let path = SyntaxPath::Mvd {
            list: s.list,
            partition: s.partition,
            axis: s.axis,
            kind: BinKind::SuffixLsb,
        };
        let key = PositionKey::new(frame_idx, mb_addr, EmbedDomain::MvdSuffixLsb, path);
        if let Some(target_bit) = injector.override_bit(key) {
            let cover_bit = suffix_lsb_bit_for_magnitude(abs);
            if target_bit != cover_bit {
                let new_abs = flipped_magnitude(abs, MVD_SUFFIX_LSB_THRESHOLD);
                s.value = if s.value < 0 {
                    -(new_abs as i32)
                } else {
                    new_abs as i32
                };
                count += 1;
            }
        }
    }
    count
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
    use crate::codec::h264::stego::PositionRecorder;

    fn luma4x4_path(coeff_idx: u8) -> SyntaxPath {
        SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx, kind: BinKind::Sign }
    }

    #[test]
    fn enumerate_empty_block() {
        let scan = vec![0i32; 16];
        let positions = enumerate_coeff_sign_positions(&scan, 0, 15, 0, 0, luma4x4_path);
        assert!(positions.is_empty());
    }

    #[test]
    fn enumerate_single_coeff() {
        let mut scan = vec![0i32; 16];
        scan[5] = 3;
        let positions = enumerate_coeff_sign_positions(&scan, 0, 15, 7, 100, luma4x4_path);
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].frame_idx(), 7);
        assert_eq!(positions[0].mb_addr(), 100);
        assert_eq!(positions[0].domain(), EmbedDomain::CoeffSignBypass);
        match positions[0].syntax_path() {
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 5, kind: BinKind::Sign } => (),
            other => panic!("wrong path {other:?}"),
        }
    }

    #[test]
    fn enumerate_reverse_scan_order() {
        let mut scan = vec![0i32; 16];
        scan[0] = 1;
        scan[3] = 2;
        scan[7] = 3;
        let positions = enumerate_coeff_sign_positions(&scan, 0, 15, 0, 0, luma4x4_path);
        // Highest scan index first (reverse scan).
        let coeff_idxs: Vec<u8> = positions
            .iter()
            .map(|k| match k.syntax_path() {
                SyntaxPath::Luma4x4 { coeff_idx, .. } => coeff_idx,
                _ => panic!(),
            })
            .collect();
        assert_eq!(coeff_idxs, vec![7, 3, 0]);
    }

    /// Hook impl: forces every position to a constant bit value.
    struct ConstantBitInjector(u8);
    impl BitInjector for ConstantBitInjector {
        fn override_bit(&mut self, _key: PositionKey) -> Option<u8> {
            Some(self.0)
        }
    }

    #[test]
    fn apply_overrides_force_all_positive() {
        let mut scan = vec![0i32; 16];
        scan[0] = 5;
        scan[2] = -3;
        scan[5] = -7;
        scan[10] = 4;
        let mut inj = ConstantBitInjector(0); // 0 = positive
        let count = apply_coeff_sign_overrides(
            &mut scan, 0, 15, 0, 0, luma4x4_path, &mut inj,
        );
        // Two were already positive, two were negative → 2 flips.
        assert_eq!(count, 2);
        assert_eq!(scan[0], 5);
        assert_eq!(scan[2], 3); // flipped
        assert_eq!(scan[5], 7); // flipped
        assert_eq!(scan[10], 4);
    }

    #[test]
    fn apply_overrides_force_all_negative() {
        let mut scan = vec![0i32; 16];
        scan[0] = 5;
        scan[2] = -3;
        scan[5] = -7;
        scan[10] = 4;
        let mut inj = ConstantBitInjector(1); // 1 = negative
        let count = apply_coeff_sign_overrides(
            &mut scan, 0, 15, 0, 0, luma4x4_path, &mut inj,
        );
        // Two were positive → 2 flips.
        assert_eq!(count, 2);
        assert_eq!(scan[0], -5); // flipped
        assert_eq!(scan[2], -3);
        assert_eq!(scan[5], -7);
        assert_eq!(scan[10], -4); // flipped
    }

    /// Hook impl: tracks every (key, bit) pair it returns. Used by
    /// tests to compare against decoder-side PositionRecorder output.
    struct PlanInjector {
        plan: std::collections::HashMap<PositionKey, u8>,
    }
    impl BitInjector for PlanInjector {
        fn override_bit(&mut self, key: PositionKey) -> Option<u8> {
            self.plan.get(&key).copied()
        }
    }

    #[test]
    fn apply_overrides_with_explicit_plan() {
        // Build a plan: only flip coeff_idx=5 to negative; leave the
        // others at their natural sign.
        let target_key = PositionKey::new(
            0, 0, EmbedDomain::CoeffSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 5, kind: BinKind::Sign },
        );
        let mut plan = std::collections::HashMap::new();
        plan.insert(target_key, 1u8);
        let mut inj = PlanInjector { plan };

        let mut scan = vec![0i32; 16];
        scan[2] = 3;
        scan[5] = 7;
        scan[10] = -4;

        let count = apply_coeff_sign_overrides(
            &mut scan, 0, 15, 0, 0, luma4x4_path, &mut inj,
        );
        // Only scan[5] should change (was 7, becomes -7).
        assert_eq!(count, 1);
        assert_eq!(scan[2], 3);
        assert_eq!(scan[5], -7);
        assert_eq!(scan[10], -4);
    }

    #[test]
    fn extract_sign_bits_matches_enumerate_order() {
        let mut scan = vec![0i32; 16];
        scan[0] = -3; // bit 1
        scan[5] = 7;  // bit 0
        scan[10] = -1; // bit 1
        let bits = extract_coeff_sign_bits(&scan, 0, 15);
        // Reverse scan order: scan[10], scan[5], scan[0]
        assert_eq!(bits, vec![1, 0, 1]);
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
    fn mvd_apply_overrides_force_negative() {
        let mut slots = vec![
            MvdSlot { list: 0, partition: 0, axis: super::super::Axis::X, value: 5 },
            MvdSlot { list: 0, partition: 0, axis: super::super::Axis::Y, value: 0 },
            MvdSlot { list: 0, partition: 1, axis: super::super::Axis::X, value: -3 },
        ];
        let mut inj = ConstantBitInjector(1);
        let count = apply_mvd_sign_overrides(&mut slots, 0, 0, &mut inj);
        assert_eq!(count, 1, "only the positive slot 0 should flip");
        assert_eq!(slots[0].value, -5);
        assert_eq!(slots[1].value, 0); // zero unchanged
        assert_eq!(slots[2].value, -3);
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
    fn domain_cover_capacity_and_total() {
        let mut cover = DomainCover::new();
        cover.coeff_sign_bypass.push(
            0,
            PositionKey::new(
                0, 0, EmbedDomain::CoeffSignBypass,
                SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 0, kind: BinKind::Sign },
            ),
        );
        cover.coeff_sign_bypass.push(
            1,
            PositionKey::new(
                0, 0, EmbedDomain::CoeffSignBypass,
                SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 1, kind: BinKind::Sign },
            ),
        );
        cover.mvd_sign_bypass.push(
            0,
            PositionKey::new(
                0, 0, EmbedDomain::MvdSignBypass,
                SyntaxPath::Mvd { list: 0, partition: 0, axis: super::super::Axis::X,
                                  kind: BinKind::Sign },
            ),
        );
        let cap = cover.capacity();
        assert_eq!(cap.coeff_sign_bypass, 2);
        assert_eq!(cap.mvd_sign_bypass, 1);
        assert_eq!(cap.coeff_suffix_lsb, 0);
        assert_eq!(cap.mvd_suffix_lsb, 0);
        assert_eq!(cover.total_len(), 3);
    }

    #[test]
    fn domain_cover_for_domain_mut_dispatches_correctly() {
        let mut cover = DomainCover::new();
        let dummy_key = PositionKey::new(
            0, 0, EmbedDomain::MvdSuffixLsb,
            SyntaxPath::Mvd { list: 0, partition: 0, axis: super::super::Axis::X,
                              kind: BinKind::SuffixLsb },
        );
        cover.for_domain_mut(EmbedDomain::MvdSuffixLsb).push(1, dummy_key);
        assert_eq!(cover.mvd_suffix_lsb.len(), 1);
        assert_eq!(cover.coeff_sign_bypass.len(), 0);
        assert_eq!(cover.mvd_sign_bypass.len(), 0);
    }

    #[test]
    fn coeff_suffix_lsb_below_threshold_not_eligible() {
        // |coeff|=15 (one below threshold) → no suffix LSB position.
        let mut scan = vec![0i32; 16];
        scan[0] = 15;
        let positions = enumerate_coeff_suffix_lsb_positions(
            &scan, 0, 15, 0, 0, luma4x4_path,
        );
        assert!(positions.is_empty());
    }

    #[test]
    fn coeff_suffix_lsb_threshold_eligible() {
        // |coeff|=16 (at threshold) → eligible. Cover bit = NOT 0 = 1.
        let mut scan = vec![0i32; 16];
        scan[0] = 16;
        let positions = enumerate_coeff_suffix_lsb_positions(
            &scan, 0, 15, 0, 0, luma4x4_path,
        );
        assert_eq!(positions.len(), 1);
        let bits = extract_coeff_suffix_lsb_bits(&scan, 0, 15);
        assert_eq!(bits, vec![1]);
    }

    #[test]
    fn coeff_suffix_lsb_threshold_flip_must_go_up() {
        // |coeff|=16: the only eligible direction is +1 → 17.
        let mut scan = vec![0i32; 16];
        scan[0] = 16;
        let mut inj = ConstantBitInjector(0); // target=0 (cover is 1)
        let count = apply_coeff_suffix_lsb_overrides(
            &mut scan, 0, 15, 0, 0, luma4x4_path, &mut inj,
        );
        assert_eq!(count, 1);
        assert_eq!(scan[0], 17, "must go +1 to stay above threshold");
        // After: cover bit at |coeff|=17 = NOT 1 = 0. ✓
    }

    #[test]
    fn coeff_suffix_lsb_above_threshold_flip_goes_down() {
        // |coeff|=20: -1 direction is preferred (toward zero).
        let mut scan = vec![0i32; 16];
        scan[0] = 20;
        let mut inj = ConstantBitInjector(0);
        // Cover at 20: NOT 0 = 1. Target=0 → flip needed.
        let count = apply_coeff_suffix_lsb_overrides(
            &mut scan, 0, 15, 0, 0, luma4x4_path, &mut inj,
        );
        assert_eq!(count, 1);
        assert_eq!(scan[0], 19, "should go -1 since |20| > threshold");
    }

    #[test]
    fn coeff_suffix_lsb_negative_sign_preserved() {
        let mut scan = vec![0i32; 16];
        scan[0] = -20;
        let mut inj = ConstantBitInjector(0);
        apply_coeff_suffix_lsb_overrides(
            &mut scan, 0, 15, 0, 0, luma4x4_path, &mut inj,
        );
        assert_eq!(scan[0], -19, "sign preserved across magnitude flip");
    }

    #[test]
    fn coeff_suffix_lsb_no_flip_when_cover_matches_target() {
        let mut scan = vec![0i32; 16];
        scan[0] = 17; // cover = NOT 1 = 0
        let mut inj = ConstantBitInjector(0); // target == cover
        let count = apply_coeff_suffix_lsb_overrides(
            &mut scan, 0, 15, 0, 0, luma4x4_path, &mut inj,
        );
        assert_eq!(count, 0);
        assert_eq!(scan[0], 17);
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

    #[test]
    fn mvd_suffix_lsb_threshold_flip_goes_up() {
        // |mvd|=9 at threshold: only +1 valid.
        let mut slots = vec![
            MvdSlot { list: 0, partition: 0, axis: super::super::Axis::X, value: 9 },
        ];
        let mut inj = ConstantBitInjector(1); // target=1, cover=0 → flip
        let count = apply_mvd_suffix_lsb_overrides(&mut slots, 0, 0, &mut inj);
        assert_eq!(count, 1);
        assert_eq!(slots[0].value, 10);
    }

    /// Deterministic LCG (no external dep) — used by the
    /// random-roundtrip property test below. Same constants as
    /// Numerical Recipes' `ranqd1`.
    struct Lcg(u32);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            self.0
        }
        fn next_bit(&mut self) -> u8 {
            (self.next_u32() & 1) as u8
        }
        fn next_bounded(&mut self, n: u32) -> u32 {
            self.next_u32() % n.max(1)
        }
        fn next_signed(&mut self, max_abs: i32) -> i32 {
            let mag = (self.next_u32() % (max_abs as u32 + 1)) as i32;
            if self.next_bit() == 0 { mag } else { -mag }
        }
    }

    /// Property test (deterministic random): for 32 LCG-seeded
    /// random scans + random injectors, verify
    /// apply_coeff_sign_overrides produces a scan whose
    /// extract_coeff_sign_bits matches the injector's plan exactly
    /// at every PositionKey.
    #[test]
    fn coeff_sign_inject_random_roundtrip_property() {
        let mut lcg = Lcg(0x1234_5678);
        for _trial in 0..32 {
            // Build a random scan with 0..=8 nonzero coefficients
            // at random positions with random signs.
            let mut scan = vec![0i32; 16];
            let nonzero_count = lcg.next_bounded(9) as usize;
            for _ in 0..nonzero_count {
                let pos = lcg.next_bounded(16) as usize;
                scan[pos] = lcg.next_signed(50);
                if scan[pos] == 0 {
                    scan[pos] = 1; // ensure nonzero so position is eligible
                }
            }

            // Build a random injector plan: per position, random bit.
            let positions = enumerate_coeff_sign_positions(
                &scan, 0, 15, 0, 0, luma4x4_path,
            );
            let plan: Vec<u8> = (0..positions.len())
                .map(|_| lcg.next_bit())
                .collect();
            let plan_map: std::collections::HashMap<PositionKey, u8> =
                positions.iter().zip(plan.iter()).map(|(&k, &b)| (k, b)).collect();

            struct PlanInjector(std::collections::HashMap<PositionKey, u8>);
            impl BitInjector for PlanInjector {
                fn override_bit(&mut self, key: PositionKey) -> Option<u8> {
                    self.0.get(&key).copied()
                }
            }
            let mut injector = PlanInjector(plan_map);

            // Capture original magnitudes for the magnitude-preservation check.
            let original_magnitudes: Vec<u32> =
                scan.iter().map(|c| c.unsigned_abs()).collect();

            apply_coeff_sign_overrides(
                &mut scan, 0, 15, 0, 0, luma4x4_path, &mut injector,
            );

            // Property 1: magnitudes preserved (Theorem 1).
            for (i, &orig) in original_magnitudes.iter().enumerate() {
                assert_eq!(
                    scan[i].unsigned_abs(),
                    orig,
                    "magnitude shifted at scan[{i}]",
                );
            }

            // Property 2: extracted sign bits match the plan.
            let extracted = extract_coeff_sign_bits(&scan, 0, 15);
            assert_eq!(
                extracted, plan,
                "extracted bits != plan after random injection",
            );
        }
    }

    /// Same property but for MVD sign overrides.
    #[test]
    fn mvd_sign_inject_random_roundtrip_property() {
        let mut lcg = Lcg(0xDEAD_BEEF);
        for _trial in 0..32 {
            // Random slot list: 1..=4 slots, mixed signs, ~25% zeros.
            let slot_count = (lcg.next_bounded(4) + 1) as usize;
            let mut slots = Vec::with_capacity(slot_count);
            for i in 0..slot_count {
                let value = if lcg.next_bounded(4) == 0 {
                    0i32
                } else {
                    lcg.next_signed(20)
                };
                slots.push(MvdSlot {
                    list: 0,
                    partition: i as u8,
                    axis: super::super::Axis::X,
                    value,
                });
            }

            let positions = enumerate_mvd_sign_positions(&slots, 0, 0);
            let plan: Vec<u8> = (0..positions.len())
                .map(|_| lcg.next_bit())
                .collect();
            let plan_map: std::collections::HashMap<PositionKey, u8> =
                positions.iter().zip(plan.iter()).map(|(&k, &b)| (k, b)).collect();
            struct PlanInjector(std::collections::HashMap<PositionKey, u8>);
            impl BitInjector for PlanInjector {
                fn override_bit(&mut self, key: PositionKey) -> Option<u8> {
                    self.0.get(&key).copied()
                }
            }
            let mut injector = PlanInjector(plan_map);

            let orig_magnitudes: Vec<u32> = slots.iter().map(|s| s.value.unsigned_abs()).collect();
            apply_mvd_sign_overrides(&mut slots, 0, 0, &mut injector);

            // Magnitudes preserved.
            for (s, &orig) in slots.iter().zip(orig_magnitudes.iter()) {
                assert_eq!(s.value.unsigned_abs(), orig);
            }
            // Extracted bits match plan.
            let extracted = extract_mvd_sign_bits(&slots);
            assert_eq!(extracted, plan);
        }
    }

    #[test]
    fn position_count_equals_extracted_bit_count() {
        let mut scan = vec![0i32; 16];
        for i in 0..16 {
            scan[i] = if i % 3 == 0 { (i as i32) - 4 } else { 0 };
        }
        let positions = enumerate_coeff_sign_positions(&scan, 0, 15, 0, 0, luma4x4_path);
        let bits = extract_coeff_sign_bits(&scan, 0, 15);
        assert_eq!(positions.len(), bits.len());
        let recorder = {
            let mut r = PositionRecorder::new();
            for pos in &positions {
                use crate::codec::h264::stego::PositionLogger;
                r.register(*pos);
            }
            r
        };
        assert_eq!(recorder.positions, positions);
    }
}
