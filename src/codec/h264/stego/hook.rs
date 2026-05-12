// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Encode-time stego hook traits + bit-packed PositionKey (Phase 6D.1).
//!
//! Two distinct trait contracts mirror the three-pass encoder model:
//! - [`PositionLogger`] is invoked during Pass 1 (GOP search). The
//!   encoder calls it for every bypass bin it would emit, in raster
//!   order. The logger decides which positions enter the cover pool
//!   and accumulates per-domain capacity counts.
//! - [`BitInjector`] is invoked during Pass 3 (entropy with
//!   injection). For each candidate position, the injector returns
//!   `Some(bit)` to override the encoder's natural bypass-bin value
//!   or `None` to leave it unchanged.
//!
//! The traits are intentionally distinct: a concrete hook
//! implementation belongs to exactly one pass, and the encoder only
//! invokes the contract that is meaningful for the pass it is in.
//! See `docs/design/video/h264/encoder-algorithms/stego-encode-time-architecture.md`
//! § 6 for rationale.
//!
//! [`PositionKey`] is a 64-bit packed integer so per-position
//! `HashMap<PositionKey, u8>` lookups in Pass 3 (∼40M positions on a
//! 1-min 1080p clip) hash via a single u64 ahash, no allocations.

/// Stego target classes — the four bypass-bin domains (Phase 6D
/// architecture decision A1, "bypass-bin-only injection").
///
/// Mapping to CAVLC pool (today's stego pipeline):
/// `CoeffSignBypass` ← `T1Sign + LevelSuffixSign`,
/// `CoeffSuffixLsb` ← `LevelSuffixMag`,
/// `MvdSignBypass` + `MvdSuffixLsb` ← `MvdLsb` (split for cleaner
/// per-domain cost modelling).
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
#[repr(u8)]
pub enum EmbedDomain {
    CoeffSignBypass = 0,
    CoeffSuffixLsb = 1,
    MvdSignBypass = 2,
    MvdSuffixLsb = 3,
}

impl EmbedDomain {
    fn from_bits(bits: u8) -> Self {
        match bits & 0x3 {
            0 => Self::CoeffSignBypass,
            1 => Self::CoeffSuffixLsb,
            2 => Self::MvdSignBypass,
            3 => Self::MvdSuffixLsb,
            _ => unreachable!(),
        }
    }
}

/// Whether a given bypass bin is the **sign** bin or the **LSB of an
/// Exp-Golomb suffix**. Folded into [`SyntaxPath`] payloads so the
/// encoder can disambiguate "this position is the sign at coeff_idx 3"
/// from "this position is the suffix LSB at coeff_idx 3".
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
#[repr(u8)]
pub enum BinKind {
    Sign = 0,
    SuffixLsb = 1,
}

impl BinKind {
    fn from_bits(bits: u8) -> Self {
        if bits & 1 == 0 { Self::Sign } else { Self::SuffixLsb }
    }
}

/// Motion-vector axis. Only meaningful inside [`SyntaxPath::Mvd`].
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
#[repr(u8)]
pub enum Axis {
    X = 0,
    Y = 1,
}

impl Axis {
    fn from_bits(bits: u8) -> Self {
        if bits & 1 == 0 { Self::X } else { Self::Y }
    }
}

/// Where in the macroblock syntax the bypass bin lives. The payload
/// fields uniquely identify a single bin within the (frame, mb)
/// position pair — together with [`EmbedDomain`] this makes
/// [`PositionKey`] globally unique.
///
/// All numeric field widths are checked against H.264 spec maxima
/// at construction; see `pack` / `unpack` for the bit layout.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum SyntaxPath {
    /// 4×4 luma residual block. block_idx ∈ 0..16, coeff_idx ∈ 0..16.
    Luma4x4 { block_idx: u8, coeff_idx: u8, kind: BinKind },
    /// 8×8 luma residual block (High Profile). block_idx ∈ 0..4,
    /// coeff_idx ∈ 0..64.
    Luma8x8 { block_idx: u8, coeff_idx: u8, kind: BinKind },
    /// Chroma AC residual. plane: 0 = Cb, 1 = Cr. block_idx ∈ 0..4,
    /// coeff_idx ∈ 0..16.
    ChromaAc { plane: u8, block_idx: u8, coeff_idx: u8, kind: BinKind },
    /// Chroma DC residual (2×2 Hadamard). plane: 0 = Cb, 1 = Cr,
    /// coeff_idx ∈ 0..4.
    ChromaDc { plane: u8, coeff_idx: u8, kind: BinKind },
    /// Luma DC residual for Intra_16x16 mode (4×4 Hadamard).
    /// coeff_idx ∈ 0..16.
    LumaDcIntra16x16 { coeff_idx: u8, kind: BinKind },
    /// Motion vector difference. list: 0 = list0, 1 = list1.
    /// partition ∈ 0..16 (covers up to 4×4 sub-MB partitions).
    Mvd { list: u8, partition: u8, axis: Axis, kind: BinKind },
}

impl SyntaxPath {
    /// Pack into 12 bits: 3-bit variant tag + 9-bit payload.
    fn pack(&self) -> u16 {
        match *self {
            SyntaxPath::Luma4x4 { block_idx, coeff_idx, kind } => {
                let p = (block_idx as u16 & 0xF)
                    | ((coeff_idx as u16 & 0xF) << 4)
                    | ((kind as u16 & 0x1) << 8);
                p << 3
            }
            SyntaxPath::Luma8x8 { block_idx, coeff_idx, kind } => {
                let p = (block_idx as u16 & 0x3)
                    | ((coeff_idx as u16 & 0x3F) << 2)
                    | ((kind as u16 & 0x1) << 8);
                1 | (p << 3)
            }
            SyntaxPath::ChromaAc { plane, block_idx, coeff_idx, kind } => {
                let p = (plane as u16 & 0x1)
                    | ((block_idx as u16 & 0x3) << 1)
                    | ((coeff_idx as u16 & 0xF) << 3)
                    | ((kind as u16 & 0x1) << 7);
                2 | (p << 3)
            }
            SyntaxPath::ChromaDc { plane, coeff_idx, kind } => {
                let p = (plane as u16 & 0x1)
                    | ((coeff_idx as u16 & 0x3) << 1)
                    | ((kind as u16 & 0x1) << 3);
                3 | (p << 3)
            }
            SyntaxPath::LumaDcIntra16x16 { coeff_idx, kind } => {
                let p = (coeff_idx as u16 & 0xF) | ((kind as u16 & 0x1) << 4);
                4 | (p << 3)
            }
            SyntaxPath::Mvd { list, partition, axis, kind } => {
                let p = (list as u16 & 0x1)
                    | ((partition as u16 & 0xF) << 1)
                    | ((axis as u16 & 0x1) << 5)
                    | ((kind as u16 & 0x1) << 6);
                5 | (p << 3)
            }
        }
    }

    /// Inverse of `pack`. Panics on invalid tag (would indicate
    /// corruption of an encoder-internal invariant).
    fn unpack(bits: u16) -> Self {
        let tag = bits & 0x7;
        let p = (bits >> 3) & 0x1FF;
        match tag {
            0 => SyntaxPath::Luma4x4 {
                block_idx: (p & 0xF) as u8,
                coeff_idx: ((p >> 4) & 0xF) as u8,
                kind: BinKind::from_bits(((p >> 8) & 0x1) as u8),
            },
            1 => SyntaxPath::Luma8x8 {
                block_idx: (p & 0x3) as u8,
                coeff_idx: ((p >> 2) & 0x3F) as u8,
                kind: BinKind::from_bits(((p >> 8) & 0x1) as u8),
            },
            2 => SyntaxPath::ChromaAc {
                plane: (p & 0x1) as u8,
                block_idx: ((p >> 1) & 0x3) as u8,
                coeff_idx: ((p >> 3) & 0xF) as u8,
                kind: BinKind::from_bits(((p >> 7) & 0x1) as u8),
            },
            3 => SyntaxPath::ChromaDc {
                plane: (p & 0x1) as u8,
                coeff_idx: ((p >> 1) & 0x3) as u8,
                kind: BinKind::from_bits(((p >> 3) & 0x1) as u8),
            },
            4 => SyntaxPath::LumaDcIntra16x16 {
                coeff_idx: (p & 0xF) as u8,
                kind: BinKind::from_bits(((p >> 4) & 0x1) as u8),
            },
            5 => SyntaxPath::Mvd {
                list: (p & 0x1) as u8,
                partition: ((p >> 1) & 0xF) as u8,
                axis: Axis::from_bits(((p >> 5) & 0x1) as u8),
                kind: BinKind::from_bits(((p >> 6) & 0x1) as u8),
            },
            _ => panic!("PositionKey: invalid SyntaxPath tag {tag}"),
        }
    }
}

/// Globally-unique stego position identifier, packed into a u64.
///
/// Layout (LSB → MSB):
/// - bits 0..23: `frame_idx` (24 bits, max 16M frames ≈ 6 days @ 30 fps)
/// - bits 24..47: `mb_addr` (24 bits, supports up to 16K×16K resolutions)
/// - bits 48..51: `domain` (4 bits, 4 used + 12 reserved)
/// - bits 52..63: `syntax_path` (12 bits — 3-bit variant + 9-bit payload)
///
/// Equality and hashing are over the raw u64. Pass 3's per-position
/// HashMap lookup is a single integer hash + comparison; no string
/// allocation, no enum match.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct PositionKey(u64);

impl PositionKey {
    /// Construct a fresh PositionKey. Field maxima are debug-asserted;
    /// release-builds silently mask out-of-range values to fit. The
    /// encoder is the sole producer and must respect the H.264 spec
    /// maxima.
    #[inline]
    pub fn new(
        frame_idx: u32,
        mb_addr: u32,
        domain: EmbedDomain,
        syntax_path: SyntaxPath,
    ) -> Self {
        debug_assert!(frame_idx < (1 << 24), "frame_idx overflow: {frame_idx}");
        debug_assert!(mb_addr < (1 << 24), "mb_addr overflow: {mb_addr}");
        let frame_bits = (frame_idx as u64) & 0xFF_FFFF;
        let mb_bits = ((mb_addr as u64) & 0xFF_FFFF) << 24;
        let domain_bits = ((domain as u64) & 0xF) << 48;
        let syntax_bits = ((syntax_path.pack() as u64) & 0xFFF) << 52;
        Self(frame_bits | mb_bits | domain_bits | syntax_bits)
    }

    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }

    #[inline]
    pub fn frame_idx(self) -> u32 {
        (self.0 & 0xFF_FFFF) as u32
    }

    #[inline]
    pub fn mb_addr(self) -> u32 {
        ((self.0 >> 24) & 0xFF_FFFF) as u32
    }

    #[inline]
    pub fn domain(self) -> EmbedDomain {
        EmbedDomain::from_bits(((self.0 >> 48) & 0xF) as u8)
    }

    #[inline]
    pub fn syntax_path(self) -> SyntaxPath {
        SyntaxPath::unpack(((self.0 >> 52) & 0xFFF) as u16)
    }
}

/// Per-GOP capacity vector, one count per [`EmbedDomain`].
/// Returned by Pass 1 to drive the Pass 2 STC plan.
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub struct GopCapacity {
    pub coeff_sign_bypass: usize,
    pub coeff_suffix_lsb: usize,
    pub mvd_sign_bypass: usize,
    pub mvd_suffix_lsb: usize,
}

impl GopCapacity {
    #[inline]
    pub fn total(&self) -> usize {
        self.coeff_sign_bypass + self.coeff_suffix_lsb
            + self.mvd_sign_bypass + self.mvd_suffix_lsb
    }

    #[inline]
    pub fn add(&mut self, domain: EmbedDomain) {
        match domain {
            EmbedDomain::CoeffSignBypass => self.coeff_sign_bypass += 1,
            EmbedDomain::CoeffSuffixLsb => self.coeff_suffix_lsb += 1,
            EmbedDomain::MvdSignBypass => self.mvd_sign_bypass += 1,
            EmbedDomain::MvdSuffixLsb => self.mvd_suffix_lsb += 1,
        }
    }
}

/// Pass 1 hook (GOP search). Encoder calls `register` for every
/// candidate bypass bin it would emit. Hook decides whether the
/// position joins the cover pool.
///
/// **Concurrency**: invoked single-threaded per GOP, but instances
/// move across rayon worker threads at GOP-parallel boundaries —
/// hence the `Send` bound. Within a GOP the architecture
/// guarantees no cross-thread access to a single instance (A7 of
/// the architecture doc, the constraint that keeps Phase I.3 row-
/// pipelining a viable post-6D.10 follow-up).
pub trait PositionLogger: Send {
    /// Returns `true` if the position is accepted into the cover pool
    /// (cost-model + per-position validity). Encoder uses the return
    /// value only for diagnostics — the encoder's own emit logic is
    /// independent of which positions are selected.
    fn register(&mut self, key: PositionKey) -> bool;

    /// Drained at end of GOP for Pass 2 STC planning.
    fn capacity(&self) -> GopCapacity;
}

/// Pass 3 hook (entropy with injection). Encoder asks whether to
/// override a bypass bin's natural value at this position.
///
/// **Concurrency**: same `Send` discipline as `PositionLogger`.
pub trait BitInjector: Send {
    /// `Some(bit)` overrides the encoder's natural bypass-bin value.
    /// `None` leaves it unchanged. The override only takes effect at
    /// positions the Pass 1 logger accepted into the cover pool.
    fn override_bit(&mut self, key: PositionKey) -> Option<u8>;
}

/// Reference [`PositionLogger`] implementation that accepts every
/// position and counts per domain. Used in Pass 1 to compute raw
/// capacity; concrete cost-aware loggers in 6D.4 will reject some
/// positions based on per-position cost.
#[derive(Default, Debug)]
pub struct PositionCounter {
    capacity: GopCapacity,
}

impl PositionCounter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PositionLogger for PositionCounter {
    #[inline]
    fn register(&mut self, key: PositionKey) -> bool {
        self.capacity.add(key.domain());
        true
    }

    #[inline]
    fn capacity(&self) -> GopCapacity {
        self.capacity
    }
}

/// Reference [`PositionLogger`] implementation that ignores every
/// position. Used by callers (typically the bin-level decoder) that
/// don't want to emit positions but are required to pass a logger
/// by the API contract.
#[derive(Default, Debug)]
pub struct NullLogger;

impl NullLogger {
    pub fn new() -> Self {
        Self
    }
}

impl PositionLogger for NullLogger {
    #[inline]
    fn register(&mut self, _key: PositionKey) -> bool {
        true
    }

    #[inline]
    fn capacity(&self) -> GopCapacity {
        GopCapacity::default()
    }
}

/// Reference [`PositionLogger`] implementation that records every
/// position into a Vec, in order. Used by tests + the 6D.3 paired
/// roundtrip validation gate to compare encoder + decoder
/// position-key sequences.
#[derive(Default, Debug)]
pub struct PositionRecorder {
    pub positions: Vec<PositionKey>,
}

impl PositionRecorder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PositionLogger for PositionRecorder {
    #[inline]
    fn register(&mut self, key: PositionKey) -> bool {
        self.positions.push(key);
        true
    }

    #[inline]
    fn capacity(&self) -> GopCapacity {
        let mut cap = GopCapacity::default();
        for key in &self.positions {
            cap.add(key.domain());
        }
        cap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(key: PositionKey) {
        let bits = key.raw();
        let restored = PositionKey(bits);
        assert_eq!(restored, key, "raw u64 roundtrip");
        // Also check accessor consistency.
        assert_eq!(restored.frame_idx(), key.frame_idx());
        assert_eq!(restored.mb_addr(), key.mb_addr());
        assert_eq!(restored.domain(), key.domain());
        assert_eq!(restored.syntax_path(), key.syntax_path());
    }

    #[test]
    fn position_key_luma4x4_roundtrip() {
        let key = PositionKey::new(
            12345,
            6789,
            EmbedDomain::CoeffSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 5, coeff_idx: 11, kind: BinKind::Sign },
        );
        roundtrip(key);
        match key.syntax_path() {
            SyntaxPath::Luma4x4 { block_idx, coeff_idx, kind } => {
                assert_eq!(block_idx, 5);
                assert_eq!(coeff_idx, 11);
                assert_eq!(kind, BinKind::Sign);
            }
            _ => panic!("wrong variant"),
        }
        assert_eq!(key.frame_idx(), 12345);
        assert_eq!(key.mb_addr(), 6789);
        assert_eq!(key.domain(), EmbedDomain::CoeffSignBypass);
    }

    #[test]
    fn position_key_luma8x8_roundtrip() {
        let key = PositionKey::new(
            0,
            8159,
            EmbedDomain::CoeffSuffixLsb,
            SyntaxPath::Luma8x8 { block_idx: 3, coeff_idx: 63, kind: BinKind::SuffixLsb },
        );
        roundtrip(key);
    }

    #[test]
    fn position_key_chroma_ac_roundtrip() {
        let key = PositionKey::new(
            999,
            12,
            EmbedDomain::CoeffSignBypass,
            SyntaxPath::ChromaAc { plane: 1, block_idx: 2, coeff_idx: 7, kind: BinKind::Sign },
        );
        roundtrip(key);
    }

    #[test]
    fn position_key_chroma_dc_roundtrip() {
        let key = PositionKey::new(
            1,
            0,
            EmbedDomain::CoeffSignBypass,
            SyntaxPath::ChromaDc { plane: 0, coeff_idx: 3, kind: BinKind::Sign },
        );
        roundtrip(key);
    }

    #[test]
    fn position_key_luma_dc_intra16_roundtrip() {
        let key = PositionKey::new(
            42,
            512,
            EmbedDomain::CoeffSignBypass,
            SyntaxPath::LumaDcIntra16x16 { coeff_idx: 15, kind: BinKind::Sign },
        );
        roundtrip(key);
    }

    #[test]
    fn position_key_mvd_roundtrip() {
        let key = PositionKey::new(
            7,
            1024,
            EmbedDomain::MvdSignBypass,
            SyntaxPath::Mvd { list: 1, partition: 15, axis: Axis::Y, kind: BinKind::Sign },
        );
        roundtrip(key);
        let key2 = PositionKey::new(
            7,
            1024,
            EmbedDomain::MvdSuffixLsb,
            SyntaxPath::Mvd { list: 0, partition: 0, axis: Axis::X, kind: BinKind::SuffixLsb },
        );
        roundtrip(key2);
        assert_ne!(key, key2, "different MVD positions must hash differently");
    }

    #[test]
    fn position_key_max_field_widths() {
        // Field maxima — exercises every bit of the packing layout.
        let key = PositionKey::new(
            (1 << 24) - 1,
            (1 << 24) - 1,
            EmbedDomain::MvdSuffixLsb,
            SyntaxPath::Luma4x4 { block_idx: 15, coeff_idx: 15, kind: BinKind::SuffixLsb },
        );
        roundtrip(key);
        assert_eq!(key.frame_idx(), (1 << 24) - 1);
        assert_eq!(key.mb_addr(), (1 << 24) - 1);
        assert_eq!(key.domain(), EmbedDomain::MvdSuffixLsb);
    }

    #[test]
    fn position_key_distinct_for_different_fields() {
        // Same (frame, mb) but different domains must give different keys.
        let base = PositionKey::new(
            10, 100, EmbedDomain::CoeffSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 0, kind: BinKind::Sign },
        );
        let alt_domain = PositionKey::new(
            10, 100, EmbedDomain::MvdSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 0, kind: BinKind::Sign },
        );
        let alt_block = PositionKey::new(
            10, 100, EmbedDomain::CoeffSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 1, coeff_idx: 0, kind: BinKind::Sign },
        );
        let alt_kind = PositionKey::new(
            10, 100, EmbedDomain::CoeffSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 0, kind: BinKind::SuffixLsb },
        );
        assert_ne!(base, alt_domain);
        assert_ne!(base, alt_block);
        assert_ne!(base, alt_kind);
        assert_ne!(alt_domain, alt_block);
    }

    #[test]
    fn embed_domain_from_bits_roundtrip() {
        for d in [
            EmbedDomain::CoeffSignBypass,
            EmbedDomain::CoeffSuffixLsb,
            EmbedDomain::MvdSignBypass,
            EmbedDomain::MvdSuffixLsb,
        ] {
            assert_eq!(EmbedDomain::from_bits(d as u8), d);
        }
    }

    #[test]
    fn position_counter_tallies_per_domain() {
        let mut counter = PositionCounter::new();
        // 5 coefficient signs, 3 mvd signs, 2 coefficient suffix LSBs.
        for i in 0..5 {
            counter.register(PositionKey::new(
                0, i, EmbedDomain::CoeffSignBypass,
                SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 0, kind: BinKind::Sign },
            ));
        }
        for i in 0..3 {
            counter.register(PositionKey::new(
                0, i, EmbedDomain::MvdSignBypass,
                SyntaxPath::Mvd { list: 0, partition: 0, axis: Axis::X, kind: BinKind::Sign },
            ));
        }
        for i in 0..2 {
            counter.register(PositionKey::new(
                0, i, EmbedDomain::CoeffSuffixLsb,
                SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 0, kind: BinKind::SuffixLsb },
            ));
        }
        let cap = counter.capacity();
        assert_eq!(cap.coeff_sign_bypass, 5);
        assert_eq!(cap.coeff_suffix_lsb, 2);
        assert_eq!(cap.mvd_sign_bypass, 3);
        assert_eq!(cap.mvd_suffix_lsb, 0);
        assert_eq!(cap.total(), 10);
    }

    #[test]
    fn position_counter_register_always_accepts() {
        let mut counter = PositionCounter::new();
        let key = PositionKey::new(
            0, 0, EmbedDomain::CoeffSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 0, kind: BinKind::Sign },
        );
        // Reference implementation accepts every position.
        assert!(counter.register(key));
    }

    /// Demonstrates the trait shape — a no-op BitInjector is trivially
    /// constructible. Concrete injection logic lands in 6D.3.
    struct NoOpInjector;
    impl BitInjector for NoOpInjector {
        fn override_bit(&mut self, _: PositionKey) -> Option<u8> {
            None
        }
    }

    #[test]
    fn bit_injector_default_is_passthrough() {
        let mut inj = NoOpInjector;
        let key = PositionKey::new(
            0, 0, EmbedDomain::CoeffSignBypass,
            SyntaxPath::Luma4x4 { block_idx: 0, coeff_idx: 0, kind: BinKind::Sign },
        );
        assert!(inj.override_bit(key).is_none());
    }

    #[test]
    fn syntax_path_pack_fits_12_bits() {
        // Worst-case payloads must not spill past bit 11.
        let cases = [
            SyntaxPath::Luma4x4 { block_idx: 15, coeff_idx: 15, kind: BinKind::SuffixLsb },
            SyntaxPath::Luma8x8 { block_idx: 3, coeff_idx: 63, kind: BinKind::SuffixLsb },
            SyntaxPath::ChromaAc { plane: 1, block_idx: 3, coeff_idx: 15, kind: BinKind::SuffixLsb },
            SyntaxPath::ChromaDc { plane: 1, coeff_idx: 3, kind: BinKind::SuffixLsb },
            SyntaxPath::LumaDcIntra16x16 { coeff_idx: 15, kind: BinKind::SuffixLsb },
            SyntaxPath::Mvd { list: 1, partition: 15, axis: Axis::Y, kind: BinKind::SuffixLsb },
        ];
        for path in cases {
            let bits = path.pack();
            assert!(bits < (1 << 12), "{path:?} packed to {bits:#x} (>12 bits)");
            assert_eq!(SyntaxPath::unpack(bits), path);
        }
    }
}
