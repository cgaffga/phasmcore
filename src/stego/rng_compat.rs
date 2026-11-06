// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! The `rand` 0.8 samplers, on top of nothing but the raw word stream.
//!
//! Every keyed layout (Ghost permutation and H-hat, shadow orders, Armor
//! STDM order and spreading vectors, Fortress, the Fortress T / TL tile,
//! the video cover order) is a function of a ChaCha20 stream *and* of how
//! `rand` turns that stream into numbers. `rand` 0.9 changed the integer
//! `gen_range` algorithm, so a bare `rand` upgrade would move every keyed
//! position of every released image and video. This module pins the
//! algorithms instead: it reproduces exactly what `rand` 0.8.5 (unchanged
//! through 0.8.8) computes, using only `next_u32` / `next_u64`, so the
//! `rand` crate can be upgraded freely. Every keyed draw in phasm goes
//! through [`Rand08`]; never call `rand`'s own samplers (`gen_range`,
//! `random_range`, `random`, `gen`, `gen_bool`, `shuffle`, ...) for anything
//! that must stay reproducible.
//!
//! Ported from `rand` 0.8.5 (`src/distributions/`):
//! - integer ranges: `uniform.rs`, `UniformInt::sample_single_inclusive`
//!   (`sample_single` is `sample_single_inclusive(low, high - 1)`): a widening
//!   multiply of one draw of the "large" type (`u32` for 8/16/32-bit types,
//!   `u64` for 64-bit, `usize` for `usize`/`isize`) by the range, keeping the
//!   high half unless the low half lies above the rejection zone. The zone is
//!   exact (`MAX - (MAX - range + 1) % range`) for 8- and 16-bit types and the
//!   approximation `(range << range.leading_zeros()) - 1` for wider ones. A
//!   range covering the whole type (wrap to 0) returns a `Standard` draw;
//! - float ranges: `uniform.rs`, `UniformFloat::sample_single`: a `[1, 2)`
//!   float from the top mantissa bits of one draw, minus 1, times
//!   `high - low`, plus `low`, redrawn while the result is not below `high`;
//! - `Standard` (`gen::<T>()`): `integer.rs`, `float.rs` (multiply method,
//!   53 / 24 bits), `other.rs` (`bool`: sign bit of `next_u32`);
//! - `gen_bool(p)`: `bernoulli.rs` (`p * 2^64` as an integer threshold
//!   against one `next_u64`; `p == 1` draws nothing);
//! - `shuffle`: `seq/mod.rs`, `SliceRandom::shuffle` (tests only).
//!
//! Pinned by `stego::rng_golden_tests` (recorded with `rand` 0.8.8) and by
//! the differential tests below against the real `rand` 0.8 (a
//! dev-dependency only).

use core::ops::{Range, RangeInclusive};

/// The raw word stream the samplers draw from (`rand_core`'s base trait,
/// `RngCore` before `rand` 0.10).
pub use rand::Rng as WordRng;

/// `rand` 0.8's sampling methods, on any [`WordRng`].
pub trait Rand08: WordRng {
    /// `rand` 0.8 `Rng::gen_range` for `a..b` and `a..=b` (integers; `a..b`
    /// for floats). Panics on an empty range, like `rand`.
    #[inline]
    fn gen_range08<T, R: SampleRange08<T>>(&mut self, range: R) -> T {
        range.sample08(self)
    }

    /// `rand` 0.8 `Rng::gen::<T>()` (the `Standard` distribution).
    #[inline]
    fn gen08<T: Standard08>(&mut self) -> T {
        T::sample08(self)
    }

    /// `rand` 0.8 `Rng::gen_bool(p)`. Panics unless `0 <= p <= 1`.
    #[inline]
    fn gen_bool08(&mut self, p: f64) -> bool {
        // 2^64, as `bernoulli.rs` writes it.
        const SCALE: f64 = 2.0 * (1u64 << 63) as f64;
        const ALWAYS_TRUE: u64 = u64::MAX;
        let p_int = if (0.0..1.0).contains(&p) {
            (p * SCALE) as u64
        } else if p == 1.0 {
            ALWAYS_TRUE
        } else {
            panic!("gen_bool08: p = {p} is not a probability");
        };
        if p_int == ALWAYS_TRUE {
            return true;
        }
        self.next_u64() < p_int
    }

    /// `rand` 0.8 `SliceRandom::shuffle` (`seq/mod.rs`): Fisher-Yates from
    /// the end, each index drawn as `u32` while the bound fits (`gen_index`).
    fn shuffle08<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let ubound = i + 1;
            let j = if ubound <= u32::MAX as usize {
                self.gen_range08(0..ubound as u32) as usize
            } else {
                self.gen_range08(0..ubound)
            };
            slice.swap(i, j);
        }
    }
}

impl<R: WordRng + ?Sized> Rand08 for R {}

/// A range [`Rand08::gen_range08`] samples from: `a..b` of a [`Uniform08`]
/// type, `a..=b` of a [`UniformInt08`] type.
pub trait SampleRange08<T> {
    fn sample08<R: WordRng + ?Sized>(self, rng: &mut R) -> T;
}

/// `rand` 0.8's `UniformSampler::sample_single` (`low..high`).
pub trait Uniform08: Sized {
    fn sample_single08<R: WordRng + ?Sized>(low: Self, high: Self, rng: &mut R) -> Self;
}

/// `rand` 0.8's `UniformSampler::sample_single_inclusive` (`low..=high`),
/// integers only.
pub trait UniformInt08: Sized {
    fn sample_single_inclusive08<R: WordRng + ?Sized>(low: Self, high: Self, rng: &mut R) -> Self;
}

impl<T: Uniform08> SampleRange08<T> for Range<T> {
    #[inline]
    fn sample08<R: WordRng + ?Sized>(self, rng: &mut R) -> T {
        T::sample_single08(self.start, self.end, rng)
    }
}

impl<T: UniformInt08> SampleRange08<T> for RangeInclusive<T> {
    #[inline]
    fn sample08<R: WordRng + ?Sized>(self, rng: &mut R) -> T {
        let (low, high) = self.into_inner();
        T::sample_single_inclusive08(low, high, rng)
    }
}

/// A type [`Rand08::gen08`] draws (`rand` 0.8's `Standard`).
pub trait Standard08: Sized {
    fn sample08<R: WordRng + ?Sized>(rng: &mut R) -> Self;
}

// ---------------------------------------------------------------------------
// Standard
// ---------------------------------------------------------------------------

impl Standard08 for u32 {
    #[inline]
    fn sample08<R: WordRng + ?Sized>(rng: &mut R) -> u32 {
        rng.next_u32()
    }
}
impl Standard08 for u64 {
    #[inline]
    fn sample08<R: WordRng + ?Sized>(rng: &mut R) -> u64 {
        rng.next_u64()
    }
}
impl Standard08 for u8 {
    #[inline]
    fn sample08<R: WordRng + ?Sized>(rng: &mut R) -> u8 {
        rng.next_u32() as u8
    }
}
impl Standard08 for u16 {
    #[inline]
    fn sample08<R: WordRng + ?Sized>(rng: &mut R) -> u16 {
        rng.next_u32() as u16
    }
}
impl Standard08 for usize {
    #[inline]
    #[cfg(any(target_pointer_width = "32", target_pointer_width = "16"))]
    fn sample08<R: WordRng + ?Sized>(rng: &mut R) -> usize {
        rng.next_u32() as usize
    }
    #[inline]
    #[cfg(target_pointer_width = "64")]
    fn sample08<R: WordRng + ?Sized>(rng: &mut R) -> usize {
        rng.next_u64() as usize
    }
}
macro_rules! standard_signed {
    ($($ty:ty => $uty:ty),*) => {$(
        impl Standard08 for $ty {
            #[inline]
            fn sample08<R: WordRng + ?Sized>(rng: &mut R) -> $ty {
                <$uty>::sample08(rng) as $ty
            }
        }
    )*};
}
standard_signed!(i8 => u8, i16 => u16, i32 => u32, i64 => u64, isize => usize);

impl Standard08 for bool {
    #[inline]
    fn sample08<R: WordRng + ?Sized>(rng: &mut R) -> bool {
        (rng.next_u32() as i32) < 0
    }
}
impl Standard08 for f64 {
    /// Multiply method, 53 bits, `[0, 1)`.
    #[inline]
    fn sample08<R: WordRng + ?Sized>(rng: &mut R) -> f64 {
        let scale = 1.0 / ((1u64 << 53) as f64);
        scale * ((rng.next_u64() >> (64 - 53)) as f64)
    }
}
impl Standard08 for f32 {
    /// Multiply method, 24 bits, `[0, 1)`.
    #[inline]
    fn sample08<R: WordRng + ?Sized>(rng: &mut R) -> f32 {
        let scale = 1.0 / ((1u32 << 24) as f32);
        scale * ((rng.next_u32() >> (32 - 24)) as f32)
    }
}

// ---------------------------------------------------------------------------
// Integer ranges
// ---------------------------------------------------------------------------

/// One draw of a "large" type and its widening multiply (`utils.rs`
/// `WideningMultiply`): `(high half, low half)` of `v * range`.
trait Large: Copy + Ord {
    const MAX: Self;
    fn draw<R: WordRng + ?Sized>(rng: &mut R) -> Self;
    fn wmul(self, range: Self) -> (Self, Self);
}
impl Large for u32 {
    const MAX: u32 = u32::MAX;
    #[inline]
    fn draw<R: WordRng + ?Sized>(rng: &mut R) -> u32 {
        rng.next_u32()
    }
    #[inline]
    fn wmul(self, range: u32) -> (u32, u32) {
        let t = u64::from(self) * u64::from(range);
        ((t >> 32) as u32, t as u32)
    }
}
impl Large for u64 {
    const MAX: u64 = u64::MAX;
    #[inline]
    fn draw<R: WordRng + ?Sized>(rng: &mut R) -> u64 {
        rng.next_u64()
    }
    #[inline]
    fn wmul(self, range: u64) -> (u64, u64) {
        let t = u128::from(self) * u128::from(range);
        ((t >> 64) as u64, t as u64)
    }
}
impl Large for usize {
    const MAX: usize = usize::MAX;
    #[inline]
    fn draw<R: WordRng + ?Sized>(rng: &mut R) -> usize {
        <usize as Standard08>::sample08(rng)
    }
    #[inline]
    #[cfg(target_pointer_width = "64")]
    fn wmul(self, range: usize) -> (usize, usize) {
        let (hi, lo) = (self as u64).wmul(range as u64);
        (hi as usize, lo as usize)
    }
    #[inline]
    #[cfg(target_pointer_width = "32")]
    fn wmul(self, range: usize) -> (usize, usize) {
        let (hi, lo) = (self as u32).wmul(range as u32);
        (hi as usize, lo as usize)
    }
}

/// `UniformInt::sample_single_inclusive` for one `($ty, $unsigned, $u_large)`
/// triple of `uniform.rs`'s `uniform_int_impl!` list.
macro_rules! int_ranges {
    ($($ty:ty, $unsigned:ty, $u_large:ty;)*) => {$(
        impl UniformInt08 for $ty {
            #[inline]
            fn sample_single_inclusive08<R: WordRng + ?Sized>(low: $ty, high: $ty, rng: &mut R) -> $ty {
                assert!(low <= high, "gen_range08: empty range");
                let range = high.wrapping_sub(low).wrapping_add(1) as $unsigned as $u_large;
                // Wrapped to 0: the range is the whole type.
                if range == 0 {
                    return <$ty as Standard08>::sample08(rng);
                }
                let zone: $u_large = if <$unsigned>::MAX as u128 <= u16::MAX as u128 {
                    // 8- and 16-bit types: the exact zone.
                    let unsigned_max = <$u_large as Large>::MAX;
                    let ints_to_reject = (unsigned_max - range + 1) % range;
                    unsigned_max - ints_to_reject
                } else {
                    // Wider types: the conservative approximation.
                    (range << range.leading_zeros()).wrapping_sub(1)
                };
                loop {
                    let v = <$u_large as Large>::draw(rng);
                    let (hi, lo) = v.wmul(range);
                    if lo <= zone {
                        return low.wrapping_add(hi as $ty);
                    }
                }
            }
        }

        impl Uniform08 for $ty {
            #[inline]
            fn sample_single08<R: WordRng + ?Sized>(low: $ty, high: $ty, rng: &mut R) -> $ty {
                assert!(low < high, "gen_range08: empty range");
                <$ty>::sample_single_inclusive08(low, high - 1, rng)
            }
        }
    )*};
}

int_ranges! {
    i8, u8, u32;
    i16, u16, u32;
    i32, u32, u32;
    i64, u64, u64;
    isize, usize, usize;
    u8, u8, u32;
    u16, u16, u32;
    u32, u32, u32;
    u64, u64, u64;
    usize, usize, usize;
}

// ---------------------------------------------------------------------------
// Float ranges
// ---------------------------------------------------------------------------

/// `UniformFloat::sample_single` (exclusive ranges; `rand` 0.8 has no
/// specialised inclusive single sample for floats, and phasm uses none).
macro_rules! float_range {
    ($ty:ty, $draw:ident, $bits_to_discard:expr, $exponent_bits:expr) => {
        impl Uniform08 for $ty {
            #[inline]
            fn sample_single08<R: WordRng + ?Sized>(low: $ty, high: $ty, rng: &mut R) -> $ty {
                assert!(low.is_finite() && high.is_finite(), "gen_range08: non-finite bound");
                assert!(low < high, "gen_range08: empty range");
                let scale = high - low;
                // rand 0.8 asserts this before its loop too ("range
                // overflow"), so its in-loop shrinking of a non-finite scale
                // never runs for finite bounds.
                assert!(scale.is_finite(), "gen_range08: range overflow");
                loop {
                    // A value in [1, 2): the draw's top bits as the mantissa.
                    let value1_2 = <$ty>::from_bits((rng.$draw() >> $bits_to_discard) | $exponent_bits);
                    let value0_1 = value1_2 - 1.0;
                    let res = value0_1 * scale + low;
                    if res < high {
                        return res;
                    }
                }
            }
        }
    };
}

float_range!(f64, next_u64, 64 - 52, 1023u64 << 52);
float_range!(f32, next_u32, 32 - 23, 127u32 << 23);

#[cfg(test)]
mod tests {
    //! Differential tests against the real `rand` 0.8 (`rand08` /
    //! `rand_chacha03`, dev-dependencies only): the same seed through
    //! `rand_chacha` 0.3 + `rand` 0.8's samplers and through the crate's
    //! `rand_chacha` + this module must give the same values.
    use super::*;
    use rand08::Rng as _;
    use rand08::RngCore as _;
    use rand08::SeedableRng as _;
    use rand::SeedableRng;

    fn pair(i: u64) -> (rand_chacha03::ChaCha20Rng, rand_chacha::ChaCha20Rng) {
        let mut s = [0u8; 32];
        for (k, b) in s.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(31).wrapping_add(k as u8).wrapping_mul(13);
        }
        s[31] = (i >> 8) as u8;
        (rand_chacha03::ChaCha20Rng::from_seed(s), rand_chacha::ChaCha20Rng::from_seed(s))
    }

    /// Bounds generator (independent of both RNGs under test).
    struct Mix(u64);
    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Interesting magnitudes: tiny, powers of two and their neighbours,
        /// around 2^31 / 2^32 / 2^63, and anything.
        fn bound(&mut self) -> u64 {
            let r = self.next();
            match r % 6 {
                0 => r >> 59,
                1 => (1u64 << (r >> 58)).wrapping_add((r >> 8) % 3).wrapping_sub(1),
                2 => (1u64 << 31).wrapping_add((r >> 8) % 1000).wrapping_sub(500),
                3 => (u32::MAX as u64).wrapping_sub((r >> 8) % 1000),
                4 => (r >> 8) % 100_000,
                _ => r,
            }
        }
    }

    macro_rules! diff_int {
        ($name:ident, $ty:ty) => {
            #[test]
            fn $name() {
                let mut m = Mix(stringify!($name).len() as u64);
                for i in 0..300 {
                    let (mut a, mut b) = pair(i);
                    for _ in 0..40 {
                        let x = m.bound() as $ty;
                        let y = m.bound() as $ty;
                        let (lo, hi) = if x <= y { (x, y) } else { (y, x) };
                        for _ in 0..8 {
                            assert_eq!(a.gen_range(lo..=hi), b.gen_range08(lo..=hi), "{lo}..={hi}");
                            if lo < hi {
                                assert_eq!(a.gen_range(lo..hi), b.gen_range08(lo..hi), "{lo}..{hi}");
                            }
                        }
                    }
                    // Full range, one value.
                    assert_eq!(a.gen_range(<$ty>::MIN..=<$ty>::MAX), b.gen_range08(<$ty>::MIN..=<$ty>::MAX));
                    assert_eq!(a.gen_range(<$ty>::MAX..=<$ty>::MAX), b.gen_range08(<$ty>::MAX..=<$ty>::MAX));
                    assert_eq!(a.gen_range(<$ty>::MIN..<$ty>::MAX), b.gen_range08(<$ty>::MIN..<$ty>::MAX));
                    // Stream position still in step.
                    assert_eq!(a.next_u64(), WordRng::next_u64(&mut b));
                }
            }
        };
    }
    diff_int!(diff_u8, u8);
    diff_int!(diff_u16, u16);
    diff_int!(diff_u32, u32);
    diff_int!(diff_u64, u64);
    diff_int!(diff_usize, usize);
    diff_int!(diff_i8, i8);
    diff_int!(diff_i16, i16);
    diff_int!(diff_i32, i32);
    diff_int!(diff_i64, i64);
    diff_int!(diff_isize, isize);

    #[test]
    fn diff_fisher_yates_u32() {
        for i in 0..20 {
            let (mut a, mut b) = pair(1000 + i);
            for k in (1..=100_000u32).rev() {
                assert_eq!(a.gen_range(0..=k), b.gen_range08(0..=k));
            }
        }
    }

    #[test]
    fn diff_float_ranges() {
        let mut m = Mix(7);
        for i in 0..200 {
            let (mut a, mut b) = pair(2000 + i);
            for _ in 0..200 {
                assert_eq!(a.gen_range(-1.0f64..1.0).to_bits(), b.gen_range08(-1.0f64..1.0).to_bits());
                assert_eq!(a.gen_range(0.0f64..1.0).to_bits(), b.gen_range08(0.0f64..1.0).to_bits());
                assert_eq!(a.gen_range(-1.0f32..1.0).to_bits(), b.gen_range08(-1.0f32..1.0).to_bits());
                // Arbitrary bounds, and ranges a few ulps wide where results
                // round up to `high` and are redrawn.
                let lo = (m.next() as i64 as f64) / (1u64 << (m.next() % 64)) as f64;
                let w = (m.next() >> 11) as f64 / (1u64 << (m.next() % 60)) as f64 + f64::MIN_POSITIVE;
                let hi = lo + w;
                if lo < hi && (hi - lo).is_finite() {
                    assert_eq!(a.gen_range(lo..hi).to_bits(), b.gen_range08(lo..hi).to_bits(), "{lo}..{hi}");
                }
                let hi = lo + lo.abs().max(1e-300) * 4.0 * f64::EPSILON;
                if lo < hi {
                    assert_eq!(a.gen_range(lo..hi).to_bits(), b.gen_range08(lo..hi).to_bits(), "{lo}..{hi}");
                }
                let (lo32, hi32) = (lo as f32, (lo as f32) + (w as f32).max(1e-30));
                if lo32 < hi32 && (hi32 - lo32).is_finite() {
                    assert_eq!(a.gen_range(lo32..hi32).to_bits(), b.gen_range08(lo32..hi32).to_bits());
                }
            }
            assert_eq!(a.next_u64(), WordRng::next_u64(&mut b));
        }
    }

    #[test]
    fn diff_standard_and_bool() {
        for i in 0..100 {
            let (mut a, mut b) = pair(3000 + i);
            for _ in 0..500 {
                assert_eq!(a.r#gen::<u8>(), b.gen08::<u8>());
                assert_eq!(a.r#gen::<u16>(), b.gen08::<u16>());
                assert_eq!(a.r#gen::<u32>(), b.gen08::<u32>());
                assert_eq!(a.r#gen::<u64>(), b.gen08::<u64>());
                assert_eq!(a.r#gen::<usize>(), b.gen08::<usize>());
                assert_eq!(a.r#gen::<i8>(), b.gen08::<i8>());
                assert_eq!(a.r#gen::<i16>(), b.gen08::<i16>());
                assert_eq!(a.r#gen::<i32>(), b.gen08::<i32>());
                assert_eq!(a.r#gen::<i64>(), b.gen08::<i64>());
                assert_eq!(a.r#gen::<isize>(), b.gen08::<isize>());
                assert_eq!(a.r#gen::<bool>(), b.gen08::<bool>());
                assert_eq!(a.r#gen::<f32>().to_bits(), b.gen08::<f32>().to_bits());
                assert_eq!(a.r#gen::<f64>().to_bits(), b.gen08::<f64>().to_bits());
                for p in [0.0, 1e-12, 0.1, 0.5, 0.75, 0.999_999, 1.0] {
                    assert_eq!(a.gen_bool(p), b.gen_bool08(p), "p = {p}");
                }
            }
            assert_eq!(a.next_u64(), WordRng::next_u64(&mut b));
        }
    }

    #[test]
    fn diff_shuffle() {
        use rand08::seq::SliceRandom as _;
        for i in 0..50u64 {
            let (mut a, mut b) = pair(5000 + i);
            for n in [0usize, 1, 2, 3, 10, 100, 1000 + i as usize] {
                let mut x: Vec<usize> = (0..n).collect();
                let mut y = x.clone();
                x.shuffle(&mut a);
                b.shuffle08(&mut y);
                assert_eq!(x, y);
            }
        }
    }

    /// The raw stream of the crate's `rand_chacha` against `rand_chacha` 0.3:
    /// `next_u32`, `next_u64` at any alignment, odd-length `fill_bytes`,
    /// `set_word_pos`, `set_stream`, `seed_from_u64` (ChaCha20 and ChaCha8).
    #[test]
    fn diff_raw_chacha_stream() {
        let mut m = Mix(11);
        for i in 0..50 {
            let (mut a, mut b) = pair(4000 + i);
            for k in 0..5000usize {
                match m.next() % 5 {
                    0 => assert_eq!(a.next_u32(), WordRng::next_u32(&mut b)),
                    1 => assert_eq!(a.next_u64(), WordRng::next_u64(&mut b)),
                    2 => {
                        let n = (m.next() % 70) as usize;
                        let (mut x, mut y) = (vec![0u8; n], vec![0u8; n]);
                        a.fill_bytes(&mut x);
                        WordRng::fill_bytes(&mut b, &mut y);
                        assert_eq!(x, y);
                    }
                    3 if k % 50 == 0 => {
                        let p = u128::from(m.next() % (1 << 40));
                        a.set_word_pos(p);
                        b.set_word_pos(p);
                        assert_eq!(a.get_word_pos(), b.get_word_pos());
                    }
                    4 if k % 500 == 0 => {
                        let s = m.next() % 4;
                        a.set_stream(s);
                        b.set_stream(s);
                    }
                    _ => assert_eq!(a.next_u32(), WordRng::next_u32(&mut b)),
                }
            }
        }
        for s in [0u64, 1, 42, 0xDEAD_BEEF, u64::MAX] {
            let mut a = rand_chacha03::ChaCha20Rng::seed_from_u64(s);
            let mut b = rand_chacha::ChaCha20Rng::seed_from_u64(s);
            let mut a8 = rand_chacha03::ChaCha8Rng::seed_from_u64(s);
            let mut b8 = rand_chacha::ChaCha8Rng::seed_from_u64(s);
            for _ in 0..1000 {
                assert_eq!(a.next_u32(), WordRng::next_u32(&mut b));
                assert_eq!(a8.next_u64(), WordRng::next_u64(&mut b8));
            }
        }
    }
}
