// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Codec-independent helpers for the tight-I420 frames the video encoders
//! take.
//!
//! Phasm writes every video upright, with no rotation tag (decided
//! 2026-10-24): a phone clip stored sideways with a display rotation is
//! turned into upright pixels before it is encoded, the way messengers,
//! HandBrake and ffmpeg transcode. Platform decoders deliver frames in
//! the coded (stored) orientation; [`rotate_i420`] turns them, in Rust, so
//! every platform rotates identically.

/// Width and height of a `width`×`height` frame after `quarter_turns_cw`
/// clockwise quarter turns.
pub fn rotated_dims(width: u32, height: u32, quarter_turns_cw: u32) -> (u32, u32) {
    if quarter_turns_cw % 2 == 1 { (height, width) } else { (width, height) }
}

/// Rotate one tight-I420 frame (Y, then U, then V; chroma planes
/// `width.div_ceil(2)` × `height.div_ceil(2)`) by `quarter_turns_cw`
/// clockwise quarter turns. The result is tight I420 at
/// [`rotated_dims`]. A lossless pixel permutation.
pub fn rotate_i420(frame: &[u8], width: u32, height: u32, quarter_turns_cw: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    assert_eq!(frame.len(), w * h + 2 * cw * ch, "not a {width}x{height} I420 frame");
    let turns = quarter_turns_cw % 4;
    if turns == 0 {
        return frame.to_vec();
    }
    let mut out = vec![0u8; frame.len()];
    let (y, uv) = frame.split_at(w * h);
    let (u, v) = uv.split_at(cw * ch);
    let (oy, ouv) = out.split_at_mut(w * h);
    let (ou, ov) = ouv.split_at_mut(cw * ch);
    rotate_plane(y, w, h, turns, oy);
    rotate_plane(u, cw, ch, turns, ou);
    rotate_plane(v, cw, ch, turns, ov);
    out
}

/// Rotate a `w`×`h` plane by 1–3 clockwise quarter turns into `dst`.
fn rotate_plane(src: &[u8], w: usize, h: usize, turns: u32, dst: &mut [u8]) {
    match turns {
        // dst is h wide: the source's bottom-left lands top-left.
        1 => {
            for (dy, row) in dst.chunks_exact_mut(h).enumerate() {
                for (dx, px) in row.iter_mut().enumerate() {
                    *px = src[(h - 1 - dx) * w + dy];
                }
            }
        }
        2 => {
            for (d, s) in dst.iter_mut().zip(src.iter().rev()) {
                *d = *s;
            }
        }
        // dst is h wide: the source's top-right lands top-left.
        3 => {
            for (dy, row) in dst.chunks_exact_mut(h).enumerate() {
                for (dx, px) in row.iter_mut().enumerate() {
                    *px = src[dx * w + (w - 1 - dy)];
                }
            }
        }
        _ => unreachable!("turns is 1..=3"),
    }
}

/// [`rotate_i420`] over a buffer of consecutive frames (a GOP).
pub fn rotate_i420_frames(
    frames: &[u8],
    width: u32,
    height: u32,
    quarter_turns_cw: u32,
) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let frame_len = w * h + 2 * w.div_ceil(2) * h.div_ceil(2);
    assert_eq!(frames.len() % frame_len, 0, "not whole {width}x{height} I420 frames");
    let mut out = Vec::with_capacity(frames.len());
    for frame in frames.chunks_exact(frame_len) {
        out.extend_from_slice(&rotate_i420(frame, width, height, quarter_turns_cw));
    }
    out
}

/// Orientation code bit: mirror left-right BEFORE the quarter turns. An
/// orientation code (0..=7) is `quarter_turns_cw | ORIENT_MIRROR?`: the
/// eight ways a video's display matrix can map stored frames to the screen.
pub const ORIENT_MIRROR: u32 = 4;

/// Width and height after orientation code `orientation` (0..=7).
pub fn oriented_dims(width: u32, height: u32, orientation: u32) -> (u32, u32) {
    rotated_dims(width, height, orientation & 3)
}

/// Turn one tight-I420 frame upright by orientation code `orientation`
/// (0..=7): mirror left-right first if [`ORIENT_MIRROR`] is set, then
/// `orientation & 3` clockwise quarter turns. Tight I420 at
/// [`oriented_dims`]; a lossless pixel permutation.
pub fn orient_i420(frame: &[u8], width: u32, height: u32, orientation: u32) -> Vec<u8> {
    assert!(orientation < 8, "orientation code {orientation} is not 0..=7");
    if orientation & ORIENT_MIRROR == 0 {
        return rotate_i420(frame, width, height, orientation);
    }
    rotate_i420(&mirror_i420(frame, width, height), width, height, orientation & 3)
}

/// [`orient_i420`] over a buffer of consecutive frames (a GOP).
pub fn orient_i420_frames(frames: &[u8], width: u32, height: u32, orientation: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let frame_len = w * h + 2 * w.div_ceil(2) * h.div_ceil(2);
    assert_eq!(frames.len() % frame_len, 0, "not whole {width}x{height} I420 frames");
    let mut out = Vec::with_capacity(frames.len());
    for frame in frames.chunks_exact(frame_len) {
        out.extend_from_slice(&orient_i420(frame, width, height, orientation));
    }
    out
}

/// Mirror every plane of a tight-I420 frame left-right.
fn mirror_i420(frame: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    assert_eq!(frame.len(), w * h + 2 * cw * ch, "not a {width}x{height} I420 frame");
    let mut out = Vec::with_capacity(frame.len());
    let (y, uv) = frame.split_at(w * h);
    for (plane, pw) in [(y, w), (uv, cw)] {
        for row in plane.chunks_exact(pw) {
            out.extend(row.iter().rev());
        }
    }
    out
}

/// Copy the top-left `width`×`height` of strided I420 planes (chroma
/// `width.div_ceil(2)`×`height.div_ceil(2)`) into one tight I420 frame —
/// the crop the streaming sessions' own packer applies, for a bridge that
/// rotates before pushing.
pub fn pack_i420(
    y: &[u8],
    y_stride: usize,
    u: &[u8],
    u_stride: usize,
    v: &[u8],
    v_stride: usize,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, crate::stego::error::StegoError> {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let fits = |plane: &[u8], stride: usize, pw: usize, ph: usize| {
        stride >= pw && (ph == 0 || plane.len() >= stride * (ph - 1) + pw)
    };
    if !fits(y, y_stride, w, h) || !fits(u, u_stride, cw, ch) || !fits(v, v_stride, cw, ch) {
        return Err(crate::stego::error::StegoError::InvalidVideo(format!(
            "I420 planes too small for {width}x{height}: strides {y_stride}/{u_stride}/{v_stride}, \
             lengths {}/{}/{}",
            y.len(),
            u.len(),
            v.len()
        )));
    }
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    for (plane, stride, pw, ph) in [(y, y_stride, w, h), (u, u_stride, cw, ch), (v, v_stride, cw, ch)] {
        for row in 0..ph {
            out.extend_from_slice(&plane[row * stride..row * stride + pw]);
        }
    }
    Ok(out)
}

/// Bytes in one tight-I420 frame of `width`×`height` (chroma planes
/// `width.div_ceil(2)`×`height.div_ceil(2)`).
fn i420_frame_len(width: u32, height: u32) -> usize {
    let (w, h) = (width as usize, height as usize);
    w * h + 2 * w.div_ceil(2) * h.div_ceil(2)
}

/// Downscale one tight-I420 frame from `src_w`×`src_h` to `dst_w`×`dst_h`
/// by area averaging: every output sample is the mean of the source area it
/// covers, with source samples it only partly covers weighted by the exact
/// fraction (1920→1280 gives each output pixel 1½ source pixels a side).
/// Each plane is scaled on its own (chroma `div_ceil(2)` of each size) and
/// rounded once, to nearest with halves up.
///
/// Pure integer arithmetic, so it is bit-identical on every platform and
/// architecture, SIMD or not — the Android app and the CLI scale with this
/// one function, so the same clip at the same preset is the same cover
/// everywhere. Equal size is a copy. Panics on a zero target, on an upscale
/// on either axis, or on a `frame` that is not one `src_w`×`src_h` frame
/// (callers validate untrusted input). For a stream of frames,
/// [`I420Scaler`] builds the weight tables once.
pub fn scale_i420(frame: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut out = Vec::new();
    I420Scaler::new(src_w, src_h, dst_w, dst_h).scale_into(frame, &mut out);
    out
}

/// [`scale_i420`] over a buffer of consecutive frames (a GOP).
pub fn scale_i420_frames(frames: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let frame_len = i420_frame_len(src_w, src_h);
    assert_eq!(frames.len() % frame_len, 0, "not whole {src_w}x{src_h} I420 frames");
    let mut scaler = I420Scaler::new(src_w, src_h, dst_w, dst_h);
    let mut out = Vec::with_capacity(frames.len() / frame_len * i420_frame_len(dst_w, dst_h));
    for frame in frames.chunks_exact(frame_len) {
        scaler.scale_into(frame, &mut out);
    }
    out
}

/// [`scale_i420`] for a stream of same-size frames: the per-axis weight
/// tables and the row scratch are built once and reused for every frame.
///
/// How it runs: per plane, for each output row the source rows it covers
/// are summed with their weights into per-column sums (a loop over the full
/// source width with one weight per row, which the compiler vectorises),
/// then each output sample sums its columns' sums with their weights and
/// divides once, by multiplying with an exact fixed-point reciprocal. That
/// column step is a gather, the bulk of the cost; it runs on NEON (aarch64)
/// or AVX2 (x86_64, detected at run time), else scalar — all three
/// bit-identical (`scaler_simd_*` tests).
pub struct I420Scaler {
    src: (u32, u32),
    dst: (u32, u32),
    luma: PlaneScaler,
    chroma: PlaneScaler,
    /// Column sums of one output row; reused across rows, planes, frames.
    cols: Vec<u32>,
}

impl I420Scaler {
    /// A scaler from `src_w`×`src_h` to `dst_w`×`dst_h`. Panics on a zero
    /// target or an upscale on either axis (callers validate untrusted
    /// input).
    pub fn new(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Self {
        Self::with_isa(src_w, src_h, dst_w, dst_h, Isa::detect())
    }

    fn with_isa(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32, isa: Isa) -> Self {
        assert!(dst_w > 0 && dst_h > 0, "cannot scale to {dst_w}x{dst_h}");
        assert!(
            dst_w <= src_w && dst_h <= src_h,
            "scale_i420 does not upscale ({src_w}x{src_h} -> {dst_w}x{dst_h})"
        );
        let (sw, sh, dw, dh) = (src_w as usize, src_h as usize, dst_w as usize, dst_h as usize);
        let half = |d: usize| d.div_ceil(2);
        Self {
            src: (src_w, src_h),
            dst: (dst_w, dst_h),
            luma: PlaneScaler::new(sw, sh, dw, dh, isa),
            chroma: PlaneScaler::new(half(sw), half(sh), half(dw), half(dh), isa),
            cols: Vec::new(),
        }
    }

    /// Size of the frames this scaler writes.
    pub fn dst_dims(&self) -> (u32, u32) {
        self.dst
    }

    /// Append `frame` (one tight-I420 frame at the source size), scaled, to
    /// `out`. Panics on a wrong-sized frame.
    pub fn scale_into(&mut self, frame: &[u8], out: &mut Vec<u8>) {
        let (sw, sh) = self.src;
        assert_eq!(frame.len(), i420_frame_len(sw, sh), "not a {sw}x{sh} I420 frame");
        out.reserve(i420_frame_len(self.dst.0, self.dst.1));
        let (y, uv) = frame.split_at(self.luma.sw * self.luma.sh);
        let (u, v) = uv.split_at(self.chroma.sw * self.chroma.sh);
        self.luma.scale(y, out, &mut self.cols);
        self.chroma.scale(u, out, &mut self.cols);
        self.chroma.scale(v, out, &mut self.cols);
    }
}

/// Which kernel runs the column step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Isa {
    Scalar,
    /// NEON on aarch64 (mandatory there), AVX2 on x86_64. Only
    /// [`Isa::detect`] returns it, and on x86_64 only after detecting AVX2:
    /// the AVX2 kernel's safety rests on that.
    Simd,
}

impl Isa {
    fn detect() -> Self {
        // NEON is mandatory on aarch64 — no runtime check.
        #[cfg(target_arch = "aarch64")]
        return Isa::Simd;
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            return Isa::Simd;
        }
        #[allow(unreachable_code)]
        Isa::Scalar
    }
}

/// One axis of an area-average downscale from `src` samples to `dst`
/// (`dst <= src`), as integer weights. In units of `1/gcd`, a source sample
/// spans `dst/gcd` and an output sample `src/gcd` ([`total`](Self::total)):
/// an output sample's weights are its overlaps with the source samples it
/// covers, and sum to `total`.
///
/// Every output sample gets the same number of taps, [`n`](Self::n) (the
/// most any needs), zero-padded, so the inner loops have a fixed length; a
/// window that would run off the end is moved left and padded in front.
/// Invariant (asserted on construction; the SIMD kernels' bounds rest on
/// it): `first[j] + n <= src` for every output `j`.
struct AxisTaps {
    src: usize,
    dst: usize,
    /// Taps per output sample.
    n: usize,
    /// First source sample of each output sample's window.
    first: Vec<u32>,
    /// Tap-major: the weight of tap `k` of output `j` is `weights[k * dst + j]`,
    /// so a SIMD kernel loads one tap of consecutive outputs contiguously.
    weights: Vec<u32>,
    total: u32,
}

impl AxisTaps {
    fn new(src: usize, dst: usize) -> Self {
        let g = gcd(src, dst);
        // u64: the products below reach src * dst.
        let (out_span, src_span) = ((src / g) as u64, (dst / g) as u64);
        let window = |j: u64| {
            let (lo, hi) = (j * out_span, (j + 1) * out_span);
            (lo, hi, lo / src_span, (hi - 1) / src_span)
        };
        let n = (0..dst as u64)
            .map(|j| {
                let (_, _, i0, i1) = window(j);
                i1 - i0 + 1
            })
            .max()
            .unwrap_or(1) as usize;
        let mut first = Vec::with_capacity(dst);
        let mut weights = vec![0u32; n * dst];
        for j in 0..dst {
            let (lo, hi, i0, i1) = window(j as u64);
            let start = (i0 as usize).min(src - n);
            first.push(start as u32);
            for i in i0..=i1 {
                let overlap = ((i + 1) * src_span).min(hi) - (i * src_span).max(lo);
                weights[(i as usize - start) * dst + j] = overlap as u32;
            }
        }
        assert!(
            first.iter().all(|&f| f as usize + n <= src),
            "scaler taps leave the {src}-sample source"
        );
        Self { src, dst, n, first, weights, total: out_span as u32 }
    }

    #[inline]
    fn weight(&self, k: usize, j: usize) -> u32 {
        self.weights[k * self.dst + j]
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// How a plane's weighted sums become samples: the sum is at most
/// `255 * den`, `den = x.total * y.total`.
enum Rounding {
    /// `2 <= den <= 4096` (nearly every real size pair: 1920x1080 → 1280x720
    /// is 9, → 848x480 is 1080): u32 sums and a 32-bit reciprocal,
    /// `((sum + den/2) * m) >> 32` with `m = floor(2^32/den) + 1` — exact
    /// while `(sum + den/2) * den < 2^32` (see [`RoundDiv::new`]), and
    /// `sum + den/2 < 256 * den`, so `den <= 4096`. The SIMD kernels run
    /// this path.
    Narrow { m: u32, half: u32 },
    /// Any larger `den` (sizes with few common factors): u64 sums, scalar.
    Wide(RoundDiv),
}

/// One plane's downscale, `sw`×`sh` → `dw`×`dh` (see [`I420Scaler`]).
struct PlaneScaler {
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    x: AxisTaps,
    y: AxisTaps,
    rounding: Rounding,
    isa: Isa,
}

impl PlaneScaler {
    fn new(sw: usize, sh: usize, dw: usize, dh: usize, isa: Isa) -> Self {
        let (x, y) = (AxisTaps::new(sw, dw), AxisTaps::new(sh, dh));
        // den is 1 only when both axes are the identity, which `scale`
        // copies before rounding is ever used.
        let den = u64::from(x.total) * u64::from(y.total);
        let rounding = if (2..=4096).contains(&den) {
            Rounding::Narrow { m: ((1u64 << 32) / den + 1) as u32, half: (den / 2) as u32 }
        } else {
            Rounding::Wide(RoundDiv::new(den))
        };
        Self { sw, sh, dw, dh, x, y, rounding, isa }
    }

    /// Append the scaled `plane` to `out`; `cols` is scratch.
    fn scale(&self, plane: &[u8], out: &mut Vec<u8>, cols: &mut Vec<u32>) {
        assert_eq!(plane.len(), self.sw * self.sh);
        if (self.sw, self.sh) == (self.dw, self.dh) {
            out.extend_from_slice(plane);
            return;
        }
        let start = out.len();
        out.resize(start + self.dw * self.dh, 0);
        cols.resize(self.sw, 0);
        for (oy, row) in out[start..].chunks_exact_mut(self.dw).enumerate() {
            self.sum_rows(plane, oy, cols);
            match &self.rounding {
                Rounding::Narrow { m, half } => columns_narrow(self.isa, &self.x, cols, row, *m, *half),
                Rounding::Wide(div) => columns_wide(&self.x, cols, row, div),
            }
        }
    }

    /// Weighted sum of the source rows output row `oy` covers, per source
    /// column, into `cols`. At most `255 * y.total` each (u32 for any
    /// height below 2^24).
    fn sum_rows(&self, plane: &[u8], oy: usize, cols: &mut [u32]) {
        let first = self.y.first[oy] as usize;
        let mut fresh = true;
        for k in 0..self.y.n {
            let w = self.y.weight(k, oy);
            if w == 0 {
                continue; // padding
            }
            let row = &plane[(first + k) * self.sw..][..self.sw];
            if fresh {
                for (c, &p) in cols.iter_mut().zip(row) {
                    *c = w * u32::from(p);
                }
                fresh = false;
            } else {
                for (c, &p) in cols.iter_mut().zip(row) {
                    *c += w * u32::from(p);
                }
            }
        }
    }
}

/// The column step on the [`Rounding::Narrow`] path: output `j` of this row
/// is `((half + Σ_k w(k, j) * cols[first[j] + k]) * m) >> 32`.
fn columns_narrow(isa: Isa, x: &AxisTaps, cols: &[u32], out: &mut [u8], m: u32, half: u32) {
    // Release-mode checks the SIMD kernels' bounds rest on, with the
    // AxisTaps invariant `first[j] + n <= src`.
    assert_eq!(cols.len(), x.src, "column sums are not one source row");
    assert_eq!(out.len(), x.dst, "output row is not one destination row");
    match x.n {
        1 => columns_narrow_n::<1>(isa, x, cols, out, m, half),
        2 => columns_narrow_n::<2>(isa, x, cols, out, m, half),
        3 => columns_narrow_n::<3>(isa, x, cols, out, m, half),
        4 => columns_narrow_n::<4>(isa, x, cols, out, m, half),
        5 => columns_narrow_n::<5>(isa, x, cols, out, m, half),
        6 => columns_narrow_n::<6>(isa, x, cols, out, m, half),
        _ => columns_narrow_scalar(x, cols, out, m, half, 0, x.n),
    }
}

/// [`columns_narrow`] at a compile-time tap count `N == x.n`: the SIMD
/// kernel takes whole blocks of 8 outputs, the scalar loop the rest.
#[inline(always)]
fn columns_narrow_n<const N: usize>(isa: Isa, x: &AxisTaps, cols: &[u32], out: &mut [u8], m: u32, half: u32) {
    assert_eq!(N, x.n);
    let done = match isa {
        Isa::Scalar => 0,
        // SAFETY: the bounds are the asserts in `columns_narrow` plus the
        // AxisTaps invariant; N == x.n is asserted above. On x86_64,
        // `Isa::Simd` exists only once AVX2 was detected; NEON is part of
        // the aarch64 baseline.
        #[cfg(target_arch = "aarch64")]
        Isa::Simd => unsafe { columns_narrow_neon::<N>(x, cols, out, m, half) },
        #[cfg(target_arch = "x86_64")]
        Isa::Simd => unsafe { columns_narrow_avx2::<N>(x, cols, out, m, half) },
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        Isa::Simd => 0,
    };
    columns_narrow_scalar(x, cols, out, m, half, done, N);
}

/// Scalar reference of the narrow column step, outputs `from..` with `n`
/// taps (`n == x.n`).
#[inline(always)]
fn columns_narrow_scalar(x: &AxisTaps, cols: &[u32], out: &mut [u8], m: u32, half: u32, from: usize, n: usize) {
    for (j, o) in out.iter_mut().enumerate().skip(from) {
        let first = x.first[j] as usize;
        let mut sum = half;
        for k in 0..n {
            sum += x.weight(k, j) * cols[first + k];
        }
        *o = ((u64::from(sum) * u64::from(m)) >> 32) as u8;
    }
}

/// The column step on the [`Rounding::Wide`] path (scalar, u64).
fn columns_wide(x: &AxisTaps, cols: &[u32], out: &mut [u8], div: &RoundDiv) {
    for (j, o) in out.iter_mut().enumerate() {
        let first = x.first[j] as usize;
        let sum: u64 = (0..x.n).map(|k| u64::from(x.weight(k, j)) * u64::from(cols[first + k])).sum();
        *o = div.div(sum);
    }
}

/// NEON column step: 8 outputs per iteration — gather each tap's column
/// sums lane by lane, multiply-accumulate with the tap's 8 contiguous
/// weights, then `(sum * m) >> 32` as a widening multiply and a narrowing
/// shift. Same integer operations as [`columns_narrow_scalar`], so the
/// same bytes. Returns how many outputs it wrote (a multiple of 8).
///
/// # Safety
/// `N == x.n`, `cols.len() == x.src`, `out.len() == x.dst`, and the
/// AxisTaps invariant `first[j] + N <= x.src`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn columns_narrow_neon<const N: usize>(x: &AxisTaps, cols: &[u32], out: &mut [u8], m: u32, half: u32) -> usize {
    use core::arch::aarch64::*;
    let dw = out.len();
    let (cp, fp, wp, op) = (cols.as_ptr(), x.first.as_ptr(), x.weights.as_ptr(), out.as_mut_ptr());
    let mut j = 0;
    // SAFETY (all pointer reads/writes below): j + 8 <= dw, so `fp.add(j + l)`
    // (l < 8) is inside `first` (len dw), `wp.add(k * dw + j + l)` (k < N) inside
    // `weights` (len N * dw), `op.add(j)..+8` inside `out`; and
    // `first[j + l] + k < first[j + l] + N <= src == cols.len()`.
    unsafe {
        let mv = vdupq_n_u32(m);
        while j + 8 <= dw {
            let mut lo = vdupq_n_u32(half);
            let mut hi = lo;
            for k in 0..N {
                let col = |l: usize| cp.add(*fp.add(j + l) as usize + k);
                let mut g0 = vdupq_n_u32(0);
                g0 = vld1q_lane_u32::<0>(col(0), g0);
                g0 = vld1q_lane_u32::<1>(col(1), g0);
                g0 = vld1q_lane_u32::<2>(col(2), g0);
                g0 = vld1q_lane_u32::<3>(col(3), g0);
                let mut g1 = vdupq_n_u32(0);
                g1 = vld1q_lane_u32::<0>(col(4), g1);
                g1 = vld1q_lane_u32::<1>(col(5), g1);
                g1 = vld1q_lane_u32::<2>(col(6), g1);
                g1 = vld1q_lane_u32::<3>(col(7), g1);
                lo = vmlaq_u32(lo, g0, vld1q_u32(wp.add(k * dw + j)));
                hi = vmlaq_u32(hi, g1, vld1q_u32(wp.add(k * dw + j + 4)));
            }
            // (sum * m) >> 32: sums < 2^20, m < 2^32, so the u64 products
            // are exact; the quotient is <= 255.
            let q_lo = vcombine_u32(
                vshrn_n_u64::<32>(vmull_u32(vget_low_u32(lo), vget_low_u32(mv))),
                vshrn_n_u64::<32>(vmull_high_u32(lo, mv)),
            );
            let q_hi = vcombine_u32(
                vshrn_n_u64::<32>(vmull_u32(vget_low_u32(hi), vget_low_u32(mv))),
                vshrn_n_u64::<32>(vmull_high_u32(hi, mv)),
            );
            vst1_u8(op.add(j), vmovn_u16(vcombine_u16(vmovn_u32(q_lo), vmovn_u32(q_hi))));
            j += 8;
        }
    }
    j
}

/// AVX2 column step: 8 outputs per iteration with a hardware gather per
/// tap; otherwise as [`columns_narrow_neon`]. Returns how many outputs it
/// wrote (a multiple of 8).
///
/// # Safety
/// AVX2 is available, `N == x.n`, `cols.len() == x.src`,
/// `out.len() == x.dst`, and the AxisTaps invariant
/// `first[j] + N <= x.src`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn columns_narrow_avx2<const N: usize>(x: &AxisTaps, cols: &[u32], out: &mut [u8], m: u32, half: u32) -> usize {
    use core::arch::x86_64::*;
    let dw = out.len();
    let (cp, fp, wp, op) = (cols.as_ptr() as *const i32, x.first.as_ptr(), x.weights.as_ptr(), out.as_mut_ptr());
    let mut j = 0;
    // SAFETY (all pointer reads/writes below): j + 8 <= dw, so the 8 `first`
    // indices at `fp.add(j)` are inside `first` (len dw), the 8 weights at
    // `wp.add(k * dw + j)` (k < N) inside `weights` (len N * dw), the 8 bytes
    // at `op.add(j)` inside `out`; every gathered index is
    // `first[j + l] + k < first[j + l] + N <= src == cols.len()`, and
    // `src < 2^31` (it is a u32 frame width that fits a u32 `first`), so the
    // i32 gather offsets do not wrap.
    unsafe {
        let mv = _mm256_set1_epi32(m as i32);
        let hv = _mm256_set1_epi32(half as i32);
        while j + 8 <= dw {
            let idx = _mm256_loadu_si256(fp.add(j) as *const __m256i);
            let mut acc = hv;
            for k in 0..N {
                let g = _mm256_i32gather_epi32::<4>(cp, _mm256_add_epi32(idx, _mm256_set1_epi32(k as i32)));
                let w = _mm256_loadu_si256(wp.add(k * dw + j) as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(g, w));
            }
            // (sum * m) >> 32 per lane: even lanes from the low halves of the
            // 64-bit products, odd lanes from the high halves.
            let even = _mm256_srli_epi64::<32>(_mm256_mul_epu32(acc, mv));
            let odd = _mm256_mul_epu32(_mm256_srli_epi64::<32>(acc), mv);
            let q = _mm256_blend_epi32::<0b1010_1010>(even, odd);
            // 8 x u32 (each <= 255) -> 8 bytes.
            let w16 = _mm_packus_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256::<1>(q));
            _mm_storel_epi64(op.add(j) as *mut __m128i, _mm_packus_epi16(w16, w16));
            j += 8;
        }
    }
    j
}

/// `round_half_up(sum / den)` for a weighted mean of 8-bit samples (`sum <=
/// 255 * den`), as a multiply by a 64-bit fixed-point reciprocal instead of
/// a division per sample. Exact (proof below); falls back to dividing for a
/// denominator too large for the reciprocal to be exact.
struct RoundDiv {
    den: u64,
    /// `floor(2^64 / den) + 1`, or 0 to divide.
    recip: u128,
}

impl RoundDiv {
    /// With `m = floor(2^B/d) + 1 = (2^B + e)/d`, `0 < e <= d`:
    /// `n*m/2^B = n/d + n*e/(d*2^B)`, and the error term is below `n/2^B`.
    /// When `n/d` is not whole it is at least `1/d` below the next integer,
    /// so the floor is exact while `n*d < 2^B`. Here `n < 256*d`, so
    /// `d <= 2^28` suffices for `B = 64` (and `d <= 4096` for the 32-bit
    /// [`Rounding::Narrow`]).
    fn new(den: u64) -> Self {
        let recip = if den <= 1 << 28 { (1u128 << 64) / u128::from(den) + 1 } else { 0 };
        Self { den, recip }
    }

    #[inline]
    fn div(&self, sum: u64) -> u8 {
        let n = sum + self.den / 2;
        if self.recip != 0 {
            ((u128::from(n) * self.recip) >> 64) as u8
        } else {
            (n / self.den) as u8
        }
    }
}

/// A per-GOP frame source that turns its frames upright, optionally
/// downscaling them first: wraps an H.264
/// [`GopYuvSource`](crate::codec::h264::openh264_stego::GopYuvSource) or
/// an AV1 [`Av1GopYuvSource`](crate::codec::av1::stego::whole_video::Av1GopYuvSource)
/// delivering `coded_width`×`coded_height` frames, scales them to the
/// [`video_target_resolve`] size ([`I420Scaler`], in the coded orientation)
/// and yields them oriented by code `orientation` ([`orient_i420`]), at
/// [`output_dims`](Self::output_dims).
// Read by the encoders' `GopYuvSource` impls below.
#[cfg_attr(not(any(feature = "h264-encoder", all(feature = "av1-encoder", feature = "av1-decoder"))), allow(dead_code))]
pub struct OrientedGopSource<S> {
    inner: S,
    width: u32,
    height: u32,
    /// Scales the coded frames before they are turned; `None` keeps the
    /// coded size. Built once, reused for every frame of every GOP.
    scaler: Option<I420Scaler>,
    orientation: u32,
}

impl<S> OrientedGopSource<S> {
    /// `orientation` must be 0..=7 (callers validate untrusted input).
    pub fn new(inner: S, coded_width: u32, coded_height: u32, orientation: u32) -> Self {
        Self::new_scaled(inner, coded_width, coded_height, None, orientation)
    }

    /// Like [`new`](Self::new), but each frame is first scaled to
    /// `scale_to` (in the coded orientation — [`VideoTarget::scale_to`]),
    /// then turned. `scale_to` must be non-zero and no larger than the
    /// coded size on either axis; `orientation` 0..=7 (callers validate
    /// untrusted input).
    pub fn new_scaled(
        inner: S,
        coded_width: u32,
        coded_height: u32,
        scale_to: Option<(u32, u32)>,
        orientation: u32,
    ) -> Self {
        assert!(orientation < 8, "orientation code {orientation} is not 0..=7");
        if let Some((w, h)) = scale_to {
            assert!(
                w > 0 && h > 0 && w <= coded_width && h <= coded_height,
                "cannot scale {coded_width}x{coded_height} to {w}x{h}: zero or upscale"
            );
        }
        // A scale to the coded size is no scale.
        let scaler = scale_to
            .filter(|&s| s != (coded_width, coded_height))
            .map(|(w, h)| I420Scaler::new(coded_width, coded_height, w, h));
        Self { inner, width: coded_width, height: coded_height, scaler, orientation }
    }

    /// Size of the frames this source yields.
    pub fn output_dims(&self) -> (u32, u32) {
        let (w, h) = self.scaler.as_ref().map_or((self.width, self.height), I420Scaler::dst_dims);
        oriented_dims(w, h, self.orientation)
    }

    /// Scale and turn one GOP of coded frames. A GOP that is not whole
    /// coded frames is an error rather than a panic (a panic aborts the
    /// mobile apps); untouched GOPs pass through as they always have.
    #[cfg_attr(not(any(feature = "h264-encoder", all(feature = "av1-encoder", feature = "av1-decoder"))), allow(dead_code))]
    fn turn(&mut self, gop: Vec<u8>) -> Result<Vec<u8>, crate::stego::error::StegoError> {
        if self.scaler.is_none() && self.orientation == 0 {
            return Ok(gop);
        }
        let (w, h) = (self.width, self.height);
        let frame_len = i420_frame_len(w, h);
        if !gop.len().is_multiple_of(frame_len) {
            return Err(crate::stego::error::StegoError::InvalidVideo(format!(
                "decoded GOP is {} bytes, not whole {w}x{h} I420 frames",
                gop.len()
            )));
        }
        let (gop, (w, h)) = match &mut self.scaler {
            Some(scaler) => {
                let (dw, dh) = scaler.dst_dims();
                let mut scaled = Vec::with_capacity(gop.len() / frame_len * i420_frame_len(dw, dh));
                for frame in gop.chunks_exact(frame_len) {
                    scaler.scale_into(frame, &mut scaled);
                }
                (scaled, (dw, dh))
            }
            None => (gop, (w, h)),
        };
        Ok(if self.orientation == 0 { gop } else { orient_i420_frames(&gop, w, h, self.orientation) })
    }
}

#[cfg(feature = "h264-encoder")]
impl<S: crate::codec::h264::openh264_stego::GopYuvSource>
    crate::codec::h264::openh264_stego::GopYuvSource for OrientedGopSource<S>
{
    fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, crate::stego::error::StegoError> {
        let gop = self.inner.gop_yuv(gop_index)?;
        self.turn(gop)
    }

    fn rewind(&mut self) -> Result<(), crate::stego::error::StegoError> {
        self.inner.rewind()
    }
}

#[cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]
impl<S: crate::codec::av1::stego::whole_video::Av1GopYuvSource>
    crate::codec::av1::stego::whole_video::Av1GopYuvSource for OrientedGopSource<S>
{
    fn gop_yuv(
        &mut self,
        gop_index: u32,
    ) -> Result<Vec<u8>, crate::codec::av1::stego::orchestrator::Av1StegoError> {
        let gop = self.inner.gop_yuv(gop_index)?;
        Ok(self.turn(gop)?)
    }

    fn rewind(&mut self) -> Result<(), crate::codec::av1::stego::orchestrator::Av1StegoError> {
        self.inner.rewind()
    }
}

/// Output size the user picks for a video encode (the HUD resolution
/// token). The FFI value is the discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoPreset {
    /// The source's own size.
    Original = 0,
    /// 480p: short edge 480 (16:9 → 848×480).
    Sd = 1,
    /// 720p: short edge 720 (16:9 → 1280×720). The default.
    Hd = 2,
    /// 1080p: short edge 1080 (16:9 → 1920×1080).
    Fhd = 3,
}

impl VideoPreset {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Original),
            1 => Some(Self::Sd),
            2 => Some(Self::Hd),
            3 => Some(Self::Fhd),
            _ => None,
        }
    }

    /// The box a scaled frame must fit, as (long edge, short edge); `None`
    /// for [`Original`](Self::Original).
    fn box_dims(self) -> Option<(u32, u32)> {
        match self {
            Self::Original => None,
            Self::Sd => Some((848, 480)),
            Self::Hd => Some((1280, 720)),
            Self::Fhd => Some((1920, 1080)),
        }
    }
}

/// Where a video encode's frames come from and what size it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoTarget {
    /// Size to scale the decoded frames to, in the source's CODED
    /// orientation (what the platform decoder delivers). `None`: keep the
    /// source size.
    pub scale_to: Option<(u32, u32)>,
    /// Orientation code that makes the frames upright ([`orient_i420`]).
    pub orientation: u32,
    /// Size of the upright frames the encoder gets: what the encode writes.
    /// Unscaled, the source floored to even (as the frame sources do).
    pub output: (u32, u32),
}

/// Resolve `preset` for a source of `coded_width`×`coded_height` that
/// displays after orientation code `orientation` (0..=7, [`orient_i420`]).
///
/// The frame is scaled to fit the preset's box — 848×480, 1280×720 or
/// 1920×1080, turned to the source's orientation — the way YouTube and
/// Apple's export presets size 480p/720p/1080p (decided 2026-10-24). The
/// edge that limits the scale lands on the box; the other keeps the
/// aspect: exactly, when that is a whole even number (squares stay square,
/// 4:5 at 720p is 720×900), else snapped to the nearest multiple of 16 as
/// messengers do (16:9 at 480p is 848×480). Clips wider than 16:9 never get
/// more pixels than the 16:9 size. A source that already fits the box is
/// kept as is: never upscale. `None` for a zero dimension.
pub fn video_target_resolve(
    coded_width: u32,
    coded_height: u32,
    orientation: u32,
    preset: VideoPreset,
) -> Option<VideoTarget> {
    if coded_width == 0 || coded_height == 0 || orientation > 7 {
        return None;
    }
    let (short_src, long_src) = if coded_width <= coded_height {
        (coded_width, coded_height)
    } else {
        (coded_height, coded_width)
    };
    let scale_to = preset.box_dims().and_then(|(box_long, box_short)| {
        if long_src <= box_long && short_src <= box_short {
            return None;
        }
        // The long edge limits when box_long / long_src <= box_short / short_src.
        let long_limits = u64::from(box_long) * u64::from(short_src)
            <= u64::from(box_short) * u64::from(long_src);
        let (long, short) = if long_limits {
            let short = derived_edge(
                u64::from(short_src) * u64::from(box_long),
                u64::from(long_src),
                box_short,
                short_src,
            );
            (box_long, short)
        } else {
            let long = derived_edge(
                u64::from(long_src) * u64::from(box_short),
                u64::from(short_src),
                box_long,
                long_src,
            );
            (long, box_short)
        };
        Some(if coded_width <= coded_height { (short, long) } else { (long, short) })
    });
    // Unscaled, the frame sources floor the source to even (I420), so
    // that is what the encode writes.
    let even = |d: u32| (d / 2 * 2).max(2);
    let (w, h) = scale_to.unwrap_or((even(coded_width), even(coded_height)));
    Some(VideoTarget {
        scale_to,
        orientation,
        output: oriented_dims(w, h, orientation),
    })
}

/// The edge that follows the aspect, `num / den`: exactly, when that is a
/// whole even number; else the nearest multiple of 16 (at least 16). Never
/// past `box_edge` or the source's `src_edge`, and even.
fn derived_edge(num: u64, den: u64, box_edge: u32, src_edge: u32) -> u32 {
    let edge = if num.is_multiple_of(den) && (num / den).is_multiple_of(2) {
        num / den
    } else {
        ((num + 8 * den) / (16 * den) * 16).max(16)
    };
    let cap = u64::from(box_edge.min(src_edge / 2 * 2).max(2));
    edge.min(cap) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve(w: u32, h: u32, turns: u32, preset: VideoPreset) -> VideoTarget {
        video_target_resolve(w, h, turns, preset).expect("valid size")
    }

    #[test]
    fn preset_values_round_trip() {
        for p in [VideoPreset::Original, VideoPreset::Sd, VideoPreset::Hd, VideoPreset::Fhd] {
            assert_eq!(VideoPreset::from_u8(p as u8), Some(p));
        }
        assert_eq!(VideoPreset::from_u8(4), None);
    }

    #[test]
    fn sixteen_by_nine_landscape_sizes() {
        let sd = resolve(1920, 1080, 0, VideoPreset::Sd);
        assert_eq!(sd, VideoTarget { scale_to: Some((848, 480)), orientation: 0, output: (848, 480) });
        assert_eq!(resolve(1920, 1080, 0, VideoPreset::Hd).scale_to, Some((1280, 720)));
        assert_eq!(resolve(3840, 2160, 0, VideoPreset::Fhd).scale_to, Some((1920, 1080)));
    }

    /// A portrait phone clip is stored landscape with a quarter turn: it is
    /// scaled in the stored orientation and comes out upright.
    #[test]
    fn rotated_portrait_scales_coded_and_outputs_upright() {
        let hd = resolve(1920, 1080, 1, VideoPreset::Hd);
        assert_eq!(hd, VideoTarget { scale_to: Some((1280, 720)), orientation: 1, output: (720, 1280) });
        // Stored upright already (no rotation): same sizes, other way round.
        let upright = resolve(1080, 1920, 0, VideoPreset::Sd);
        assert_eq!(upright.scale_to, Some((480, 848)));
        assert_eq!(upright.output, (480, 848));
    }

    #[test]
    fn four_by_three_sizes() {
        assert_eq!(resolve(1440, 1080, 0, VideoPreset::Sd).scale_to, Some((640, 480)));
        assert_eq!(resolve(1440, 1080, 0, VideoPreset::Hd).scale_to, Some((960, 720)));
    }

    #[test]
    fn never_upscales() {
        // Short edge at or below the preset's: the source size, unscaled.
        let t = resolve(1280, 720, 1, VideoPreset::Hd);
        assert_eq!(t, VideoTarget { scale_to: None, orientation: 1, output: (720, 1280) });
        assert_eq!(resolve(640, 360, 0, VideoPreset::Fhd).scale_to, None);
        assert_eq!(resolve(1920, 1080, 0, VideoPreset::Fhd).scale_to, None);
        // Original never scales.
        assert_eq!(resolve(3840, 2160, 2, VideoPreset::Original).scale_to, None);
        assert_eq!(resolve(3840, 2160, 2, VideoPreset::Original).output, (3840, 2160));
    }

    /// Squares stay square at every preset, including 1080 (not a multiple
    /// of 16).
    #[test]
    fn squares_stay_square() {
        assert_eq!(resolve(1440, 1440, 0, VideoPreset::Sd).scale_to, Some((480, 480)));
        assert_eq!(resolve(1440, 1440, 0, VideoPreset::Hd).scale_to, Some((720, 720)));
        assert_eq!(resolve(1440, 1440, 0, VideoPreset::Fhd).scale_to, Some((1080, 1080)));
        assert_eq!(resolve(1080, 1080, 0, VideoPreset::Fhd).scale_to, None);
    }

    /// An exact even size is kept rather than snapped to 16.
    #[test]
    fn four_by_five_keeps_its_exact_size() {
        assert_eq!(resolve(1080, 1350, 0, VideoPreset::Sd).scale_to, Some((480, 600)));
        assert_eq!(resolve(1080, 1350, 0, VideoPreset::Hd).scale_to, Some((720, 900)));
    }

    /// Wider than 16:9: the long edge is the box's, so the clip never gets
    /// more pixels than the 16:9 size of the same preset.
    #[test]
    fn ultrawide_fits_the_box() {
        assert_eq!(resolve(2560, 1080, 0, VideoPreset::Sd).scale_to, Some((848, 352)));
        assert_eq!(resolve(2560, 1080, 0, VideoPreset::Hd).scale_to, Some((1280, 540)));
        assert_eq!(resolve(2560, 1080, 0, VideoPreset::Fhd).scale_to, Some((1920, 810)));
        assert_eq!(resolve(3840, 1608, 0, VideoPreset::Hd).scale_to, Some((1280, 536)));
        assert_eq!(resolve(3840, 1608, 0, VideoPreset::Fhd).scale_to, Some((1920, 804)));
    }

    /// Snapping the derived edge to 16 must not leave the box or the source.
    #[test]
    fn derived_edge_stays_within_box_and_source() {
        // 481 * 848 / 860 = 474.3 → 480, the box's short edge.
        assert_eq!(resolve(860, 481, 0, VideoPreset::Sd).scale_to, Some((848, 480)));
        // Nearly square: short-limited, long 489.8 → 496.
        assert_eq!(resolve(500, 490, 0, VideoPreset::Sd).scale_to, Some((496, 480)));
        // An absurd panorama: the derived edge never exceeds the source's.
        let t = resolve(4000, 6, 0, VideoPreset::Sd).scale_to.unwrap();
        assert!(t.0 == 848 && t.1 <= 6 && t.1.is_multiple_of(2), "{t:?}");
    }

    /// Over a grid of source sizes, presets and turns: a scaled target fits
    /// the preset's box (turned to the source's orientation) with one edge
    /// on the box; the other edge is the exact even size or the nearest
    /// multiple of 16, within the source; nothing is upscaled or flipped;
    /// the output is the scaled size turned upright.
    #[test]
    fn invariants_over_source_sizes() {
        let presets = [VideoPreset::Original, VideoPreset::Sd, VideoPreset::Hd, VideoPreset::Fhd];
        for w in (2..=4000u32).step_by(58) {
            for h in (2..=4000u32).step_by(62) {
                for preset in presets {
                    for orientation in 0..8 {
                        let t = resolve(w, h, orientation, preset);
                        let (sw, sh) = t.scale_to.unwrap_or(((w / 2 * 2).max(2), (h / 2 * 2).max(2)));
                        assert_eq!(t.output, oriented_dims(sw, sh, orientation));
                        let (l, s) = (w.max(h), w.min(h));
                        let Some((sw, sh)) = t.scale_to else {
                            assert!(
                                preset.box_dims().is_none_or(|(bl, bs)| l <= bl && s <= bs),
                                "{w}x{h} {preset:?}: kept the source size but it exceeds the box"
                            );
                            continue;
                        };
                        let (bl, bs) = preset.box_dims().unwrap();
                        let (tl, ts) = (sw.max(sh), sw.min(sh));
                        let ctx = format!("{w}x{h} {preset:?} -> {sw}x{sh}");
                        assert!(l > bl || s > bs, "{ctx}: scaled a source that fits");
                        assert!(tl <= bl && ts <= bs, "{ctx}: outside the box");
                        assert!(tl <= l && ts <= s, "{ctx}: upscaled");
                        // One edge on the box; the other derived from the aspect.
                        // The limiting edge (smaller scale factor) is on the
                        // box; the derived one may snap onto it too.
                        let long_limits = u64::from(bl) * u64::from(s) <= u64::from(bs) * u64::from(l);
                        let (derived, exact, src_edge) = if long_limits {
                            assert_eq!(tl, bl, "{ctx}: limiting long edge not on the box");
                            (ts, f64::from(s) * f64::from(bl) / f64::from(l), s)
                        } else {
                            assert_eq!(ts, bs, "{ctx}: limiting short edge not on the box");
                            (tl, f64::from(l) * f64::from(bs) / f64::from(s), l)
                        };
                        let exact_even = (exact - exact.round()).abs() < 1e-9
                            && (exact.round() as u32).is_multiple_of(2);
                        if exact_even {
                            assert_eq!(f64::from(derived), exact.round(), "{ctx}: exact size not kept");
                        } else {
                            assert!(
                                derived % 16 == 0 && (f64::from(derived) - exact).abs() <= 8.0
                                    || derived <= 16
                                    || derived == src_edge / 2 * 2,
                                "{ctx}: derived edge {derived} vs exact {exact}"
                            );
                        }
                        assert!(
                            !(sw > sh && w < h || sw < sh && w > h),
                            "{ctx}: orientation flipped"
                        );
                    }
                }
            }
        }
    }

    /// Padded strides are cropped to the frame; the result is the tight
    /// frame the strides came from.
    #[test]
    fn pack_crops_strided_planes_to_tight() {
        let (w, h) = (6u32, 4u32);
        let tight = numbered_frame(w, h, 5);
        let (cw, ch) = (3usize, 2usize);
        let pad = |plane: &[u8], pw: usize, stride: usize| -> Vec<u8> {
            plane.chunks(pw).flat_map(|row| {
                let mut r = row.to_vec();
                r.resize(stride, 0xEE);
                r
            }).collect()
        };
        let (ty, tuv) = tight.split_at(24);
        let (tu, tv) = tuv.split_at(cw * ch);
        let (y, u, v) = (pad(ty, 6, 8), pad(tu, cw, 5), pad(tv, cw, 4));
        assert_eq!(pack_i420(&y, 8, &u, 5, &v, 4, w, h).unwrap(), tight);
        // Tight strides are a plain copy.
        assert_eq!(pack_i420(ty, 6, tu, cw, tv, cw, w, h).unwrap(), tight);
    }

    #[test]
    fn pack_rejects_short_strides_and_planes() {
        let y = vec![0u8; 8 * 4];
        let c = vec![0u8; 4 * 2];
        assert!(pack_i420(&y, 5, &c, 4, &c, 4, 6, 4).is_err(), "luma stride below width");
        assert!(pack_i420(&y, 8, &c, 2, &c, 4, 6, 4).is_err(), "chroma stride below width/2");
        assert!(pack_i420(&y[..20], 8, &c, 4, &c, 4, 6, 4).is_err(), "luma plane too short");
        assert!(pack_i420(&y, 8, &c[..3], 4, &c, 4, 6, 4).is_err(), "chroma plane too short");
    }

    /// Two GOPs of 2 frames plus a trailing 1-frame GOP, 6x4.
    #[cfg(any(feature = "h264-encoder", all(feature = "av1-encoder", feature = "av1-decoder")))]
    fn three_gop_clip() -> (Vec<u8>, u32, u32, u32, u32) {
        let (w, h, n, gop) = (6u32, 4u32, 5u32, 2u32);
        let clip = (0..n).flat_map(|f| numbered_frame(w, h, f as u8 * 31)).collect();
        (clip, w, h, n, gop)
    }

    #[cfg(feature = "h264-encoder")]
    #[test]
    fn oriented_h264_source_turns_each_gop() {
        use crate::codec::h264::openh264_stego::{GopYuvSource, SliceYuvSource};
        let (clip, w, h, n, gop) = three_gop_clip();
        let frame = i420_len(w, h);
        let mut plain = SliceYuvSource::new(&clip, w, h, n, gop);
        let mut src = OrientedGopSource::new(SliceYuvSource::new(&clip, w, h, n, gop), w, h, 1);
        assert_eq!(src.output_dims(), (4, 6));
        src.rewind().unwrap();
        plain.rewind().unwrap();
        for g in 0..3 {
            let got = src.gop_yuv(g).unwrap();
            assert_eq!(got, rotate_i420_frames(&plain.gop_yuv(g).unwrap(), w, h, 1), "GOP {g}");
            assert_eq!(got.len(), if g < 2 { 2 * frame } else { frame });
        }
        assert!(src.gop_yuv(3).unwrap().is_empty(), "end of stream passes through");
    }

    #[cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]
    #[test]
    fn oriented_av1_source_mirrors_and_turns_each_gop_and_rewinds() {
        use crate::codec::av1::stego::whole_video::{Av1GopYuvSource, Av1SliceYuvSource};
        let (clip, w, h, n, gop) = three_gop_clip();
        let mut plain = Av1SliceYuvSource::new(&clip, w, h, n, gop);
        let code = ORIENT_MIRROR | 1;
        let mut src = OrientedGopSource::new(Av1SliceYuvSource::new(&clip, w, h, n, gop), w, h, code);
        assert_eq!(src.output_dims(), (4, 6));
        // The rewind contract reaches the inner source.
        assert!(src.gop_yuv(0).is_err(), "gop_yuv before rewind must fail as the inner source does");
        src.rewind().unwrap();
        plain.rewind().unwrap();
        for g in 0..3 {
            assert_eq!(
                src.gop_yuv(g).unwrap(),
                orient_i420_frames(&plain.gop_yuv(g).unwrap(), w, h, code),
                "GOP {g}"
            );
        }
        assert!(src.gop_yuv(3).unwrap().is_empty());
    }

    #[cfg(feature = "h264-encoder")]
    #[test]
    fn oriented_source_with_code_zero_is_the_inner_source() {
        use crate::codec::h264::openh264_stego::{GopYuvSource, SliceYuvSource};
        let (clip, w, h, n, gop) = three_gop_clip();
        let mut plain = SliceYuvSource::new(&clip, w, h, n, gop);
        let mut src = OrientedGopSource::new(SliceYuvSource::new(&clip, w, h, n, gop), w, h, 0);
        assert_eq!(src.output_dims(), (w, h));
        for g in 0..4 {
            assert_eq!(src.gop_yuv(g).unwrap(), plain.gop_yuv(g).unwrap(), "GOP {g}");
        }
    }

    /// The output is what the encode writes: frame sources floor the
    /// source to even, so an odd source's Original reports the even size.
    #[test]
    fn original_output_is_floored_to_even() {
        let t = resolve(1919, 1081, 0, VideoPreset::Original);
        assert_eq!(t, VideoTarget { scale_to: None, orientation: 0, output: (1918, 1080) });
        assert_eq!(resolve(1081, 1919, 1, VideoPreset::Hd).output, (1280, 720));
        assert_eq!(resolve(721, 401, 0, VideoPreset::Hd).output, (720, 400));
    }

    #[test]
    fn zero_size_or_bad_orientation_is_rejected() {
        assert_eq!(video_target_resolve(1920, 1080, 8, VideoPreset::Hd), None);
        assert_eq!(video_target_resolve(0, 1080, 0, VideoPreset::Hd), None);
        assert_eq!(video_target_resolve(1920, 0, 0, VideoPreset::Original), None);
    }

    fn i420_len(w: u32, h: u32) -> usize {
        (w * h) as usize + 2 * (w.div_ceil(2) * h.div_ceil(2)) as usize
    }

    /// A frame whose every byte differs from its neighbours, so any wrong
    /// permutation shows.
    fn numbered_frame(w: u32, h: u32, seed: u8) -> Vec<u8> {
        (0..i420_len(w, h)).map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed)).collect()
    }

    #[test]
    fn dims_swap_on_odd_quarter_turns() {
        assert_eq!(rotated_dims(1920, 1080, 0), (1920, 1080));
        assert_eq!(rotated_dims(1920, 1080, 1), (1080, 1920));
        assert_eq!(rotated_dims(1920, 1080, 2), (1920, 1080));
        assert_eq!(rotated_dims(1920, 1080, 3), (1080, 1920));
        assert_eq!(rotated_dims(1920, 1080, 5), (1080, 1920));
    }

    #[test]
    fn zero_turns_is_identity() {
        let f = numbered_frame(6, 4, 1);
        assert_eq!(rotate_i420(&f, 6, 4, 0), f);
    }

    /// 4x2 luma, 2x1 chroma, turned a quarter clockwise: the bottom-left
    /// sample goes to the top-left, the top-left to the top-right.
    #[test]
    fn quarter_turn_clockwise_exact() {
        #[rustfmt::skip]
        let frame = [
            1, 2, 3, 4,
            5, 6, 7, 8,   // Y 4x2
            10, 11,       // U 2x1
            20, 21,       // V 2x1
        ];
        #[rustfmt::skip]
        let expected = [
            5, 1,
            6, 2,
            7, 3,
            8, 4,         // Y 2x4
            10,
            11,           // U 1x2
            20,
            21,           // V 1x2
        ];
        assert_eq!(rotate_i420(&frame, 4, 2, 1), expected);
    }

    #[test]
    fn half_turn_reverses_each_plane() {
        #[rustfmt::skip]
        let frame = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 20, 21];
        #[rustfmt::skip]
        let expected = [8, 7, 6, 5, 4, 3, 2, 1, 11, 10, 21, 20];
        assert_eq!(rotate_i420(&frame, 4, 2, 2), expected);
    }

    #[test]
    fn turns_compose_and_four_is_identity() {
        // Odd sizes too: the chroma planes round up.
        for (w, h) in [(6u32, 4u32), (5, 3), (16, 9), (1, 7)] {
            let f = numbered_frame(w, h, 3);
            let one = rotate_i420(&f, w, h, 1);
            let (w1, h1) = rotated_dims(w, h, 1);
            let two = rotate_i420(&one, w1, h1, 1);
            assert_eq!(two, rotate_i420(&f, w, h, 2), "{w}x{h}: 1+1 != 2");
            let (w2, h2) = rotated_dims(w, h, 2);
            let three = rotate_i420(&two, w2, h2, 1);
            assert_eq!(three, rotate_i420(&f, w, h, 3), "{w}x{h}: 1+1+1 != 3");
            let (w3, h3) = rotated_dims(w, h, 3);
            assert_eq!(rotate_i420(&three, w3, h3, 1), f, "{w}x{h}: four turns");
        }
    }

    #[test]
    fn mirror_reverses_each_row() {
        #[rustfmt::skip]
        let frame = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 20, 21];
        #[rustfmt::skip]
        let expected = [4, 3, 2, 1, 8, 7, 6, 5, 11, 10, 21, 20];
        assert_eq!(orient_i420(&frame, 4, 2, ORIENT_MIRROR), expected);
        assert_eq!(oriented_dims(4, 2, ORIENT_MIRROR), (4, 2));
    }

    /// Mirror first, then the quarter turn: [[1 2 3 4][5 6 7 8]] mirrored is
    /// [[4 3 2 1][8 7 6 5]], turned clockwise its bottom-left (8) lands
    /// top-left.
    #[test]
    fn mirror_then_quarter_turn_exact() {
        #[rustfmt::skip]
        let frame = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 20, 21];
        #[rustfmt::skip]
        let expected = [
            8, 4,
            7, 3,
            6, 2,
            5, 1,
            11, 10,
            21, 20,
        ];
        assert_eq!(orient_i420(&frame, 4, 2, ORIENT_MIRROR | 1), expected);
        assert_eq!(oriented_dims(4, 2, ORIENT_MIRROR | 1), (2, 4));
    }

    #[test]
    fn plain_codes_are_the_rotations() {
        let f = numbered_frame(6, 4, 9);
        for turns in 0..4 {
            assert_eq!(orient_i420(&f, 6, 4, turns), rotate_i420(&f, 6, 4, turns));
            assert_eq!(oriented_dims(6, 4, turns), rotated_dims(6, 4, turns));
        }
    }

    /// The eight codes are the eight distinct symmetries of a rectangle's
    /// pixel grid; a mirrored code undoes itself, a turn by its complement.
    #[test]
    fn eight_distinct_codes_with_inverses() {
        for (w, h) in [(6u32, 4u32), (5, 3), (4, 4)] {
            let f = numbered_frame(w, h, 1);
            let mut seen = Vec::new();
            for code in 0..8 {
                let once = orient_i420(&f, w, h, code);
                let (w1, h1) = oriented_dims(w, h, code);
                if (w, h) != (4, 4) || !seen.contains(&once) {
                    assert!(!seen.contains(&once), "{w}x{h}: code {code} repeats another");
                }
                seen.push(once.clone());
                let inverse = if code & ORIENT_MIRROR != 0 { code } else { (4 - code) % 4 };
                assert_eq!(orient_i420(&once, w1, h1, inverse), f, "{w}x{h}: code {code}");
            }
        }
    }

    #[test]
    fn frames_orient_one_by_one() {
        let (w, h) = (6u32, 4u32);
        let a = numbered_frame(w, h, 1);
        let b = numbered_frame(w, h, 9);
        let both = [a.clone(), b.clone()].concat();
        let code = ORIENT_MIRROR | 3;
        let expected = [orient_i420(&a, w, h, code), orient_i420(&b, w, h, code)].concat();
        assert_eq!(orient_i420_frames(&both, w, h, code), expected);
    }

    #[test]
    fn frames_rotate_one_by_one() {
        let (w, h) = (6u32, 4u32);
        let a = numbered_frame(w, h, 1);
        let b = numbered_frame(w, h, 9);
        let both = [a.clone(), b.clone()].concat();
        let expected = [rotate_i420(&a, w, h, 3), rotate_i420(&b, w, h, 3)].concat();
        assert_eq!(rotate_i420_frames(&both, w, h, 3), expected);
        assert!(rotate_i420_frames(&[], w, h, 1).is_empty());
    }

    // ─── scale_i420 ───────────────────────────────────────────────

    /// Brute-force area average, independent of the implementation: split
    /// every source sample into `dw`×`dh` cells, so each output sample is
    /// exactly `sw`×`sh` cells; average them and round half up.
    fn reference_scale_plane(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(dw * dh);
        for oy in 0..dh {
            for ox in 0..dw {
                let mut sum = 0u64;
                for cy in oy * sh..(oy + 1) * sh {
                    for cx in ox * sw..(ox + 1) * sw {
                        sum += u64::from(src[(cy / dh) * sw + cx / dw]);
                    }
                }
                let n = (sw * sh) as u64;
                out.push(((sum + n / 2) / n) as u8);
            }
        }
        out
    }

    /// [`reference_scale_plane`] over the three planes of a tight I420 frame.
    fn reference_scale_i420(frame: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
        let (sw, sh, dw, dh) = (sw as usize, sh as usize, dw as usize, dh as usize);
        let (scw, sch, dcw, dch) = (sw.div_ceil(2), sh.div_ceil(2), dw.div_ceil(2), dh.div_ceil(2));
        let (y, uv) = frame.split_at(sw * sh);
        let (u, v) = uv.split_at(scw * sch);
        let mut out = reference_scale_plane(y, sw, sh, dw, dh);
        out.extend(reference_scale_plane(u, scw, sch, dcw, dch));
        out.extend(reference_scale_plane(v, scw, sch, dcw, dch));
        out
    }

    /// A frame with distinct-ish, irregular samples in every plane.
    fn noisy_frame(w: u32, h: u32, seed: u32) -> Vec<u8> {
        (0..i420_len(w, h) as u32)
            .map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(40_503)) >> 13) as u8)
            .collect()
    }

    #[test]
    fn scale_to_the_same_size_is_a_copy() {
        for (w, h) in [(6u32, 4u32), (5, 3), (1, 1), (16, 9)] {
            let f = numbered_frame(w, h, 2);
            assert_eq!(scale_i420(&f, w, h, w, h), f, "{w}x{h}");
        }
    }

    /// 4x4 → 2x2: every output sample is its 2x2 block's mean, rounded to
    /// nearest with halves up; the 2x2 chroma planes go to 1x1.
    #[test]
    fn scale_two_to_one_averages_each_block_exactly() {
        #[rustfmt::skip]
        let frame = [
            1, 2,     0, 0,
            5, 6,     0, 1,       // blocks: (1+2+5+6)/4 = 3.5 → 4; 1/4 → 0
            10, 10,   200, 201,
            10, 11,   202, 203,   // 41/4 = 10.25 → 10; 806/4 = 201.5 → 202
            7, 8, 9, 9,           // U 2x2: 33/4 = 8.25 → 8
            255, 255, 254, 255,   // V 2x2: 1019/4 = 254.75 → 255
        ];
        let expected = [4, 0, 10, 202, 8, 255];
        assert_eq!(scale_i420(&frame, 4, 4, 2, 2), expected);
    }

    /// 6 → 4 is a 3:2 ratio: each output sample covers one and a half
    /// source samples, so the boundary sample counts half.
    #[test]
    fn scale_three_to_two_weights_the_shared_sample_by_half() {
        // Y 6x2, U and V 3x1. Rows are identical, so the vertical axis is
        // the identity and the horizontal weights show.
        #[rustfmt::skip]
        let frame = [
            30, 60, 90, 0, 3, 99,
            30, 60, 90, 0, 3, 99,
            9, 0, 30,
            100, 200, 50,
        ];
        // Y: (2*30+60)/3 = 40, (60+2*90)/3 = 80, (2*0+3)/3 = 1, (3+2*99)/3 = 67.
        // Chroma 3 → 2: (2a+b)/3, (b+2c)/3.
        #[rustfmt::skip]
        let expected = [
            40, 80, 1, 67,
            40, 80, 1, 67,
            6, 20,                 // (18+0)/3 = 6, (0+60)/3 = 20
            133, 100,              // 400/3 = 133.3 → 133, 300/3 = 100
        ];
        assert_eq!(scale_i420(&frame, 6, 2, 4, 2), expected);
    }

    /// Exact against the brute-force area average over a spread of size
    /// pairs: integer and non-integer ratios, odd sizes (chroma rounds up),
    /// one axis only, and the small analogues of 1920→1280, 1080→720 and
    /// 1920x1080→848x480.
    #[test]
    fn scale_matches_the_exact_area_average() {
        let pairs: &[((u32, u32), (u32, u32))] = &[
            ((12, 8), (8, 8)),
            ((12, 9), (8, 6)),
            ((24, 18), (16, 12)),
            ((96, 54), (53, 30)),
            ((7, 5), (5, 3)),
            ((9, 9), (4, 4)),
            ((13, 7), (13, 2)),
            ((10, 10), (1, 1)),
            ((33, 17), (32, 16)),
            ((64, 36), (42, 24)),
            // 3.5:1 → 4 taps per output, on both axes and in chroma.
            ((14, 7), (4, 2)),
            // 2.5:1, chroma 3:1 — 3 taps.
            ((10, 6), (4, 2)),
            // Coprime sizes: den 71 * 67 > 4096, the wide (u64) rounding path.
            ((71, 67), (70, 66)),
        ];
        for (seed, &((sw, sh), (dw, dh))) in pairs.iter().enumerate() {
            let f = noisy_frame(sw, sh, seed as u32);
            let reference = reference_scale_i420(&f, sw, sh, dw, dh);
            assert_eq!(scale_i420(&f, sw, sh, dw, dh), reference, "{sw}x{sh} -> {dw}x{dh}");
            let mut scalar = Vec::new();
            I420Scaler::with_isa(sw, sh, dw, dh, Isa::Scalar).scale_into(&f, &mut scalar);
            assert_eq!(scalar, reference, "{sw}x{sh} -> {dw}x{dh}, scalar kernel");
        }
    }

    /// xorshift32: deterministic test randomness.
    fn next_rand(state: &mut u32) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state
    }

    /// Scale `frame` with the detected kernel (NEON / AVX2 when present) and
    /// with the scalar one; both must give the same bytes.
    fn assert_simd_matches_scalar(frame: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) {
        let (mut simd, mut scalar) = (Vec::new(), Vec::new());
        I420Scaler::with_isa(sw, sh, dw, dh, Isa::detect()).scale_into(frame, &mut simd);
        I420Scaler::with_isa(sw, sh, dw, dh, Isa::Scalar).scale_into(frame, &mut scalar);
        assert_eq!(simd, scalar, "{sw}x{sh} -> {dw}x{dh}: SIMD differs from scalar");
    }

    /// SIMD column kernels against the scalar one over random sizes and
    /// ratios: odd widths, every tail length after the 8-wide blocks, 1 to
    /// 6+ taps, and both rounding paths. On x86_64 without AVX2 this
    /// compares scalar with scalar (CI runs it on AVX2 and under Miri —
    /// `.github/workflows/simd-parity.yml`).
    #[test]
    fn scaler_simd_matches_scalar_over_random_sizes() {
        let mut state = 0x9E37_79B9u32;
        let cases = if cfg!(miri) { 6 } else { 400 };
        let (mut narrow, mut wide) = (0, 0);
        for _ in 0..cases {
            let sw = 1 + next_rand(&mut state) % 90;
            let sh = 1 + next_rand(&mut state) % 30;
            let dw = 1 + next_rand(&mut state) % sw;
            let dh = 1 + next_rand(&mut state) % sh;
            let scaler = I420Scaler::new(sw, sh, dw, dh);
            for plane in [&scaler.luma, &scaler.chroma] {
                match plane.rounding {
                    Rounding::Narrow { .. } => narrow += 1,
                    Rounding::Wide(_) => wide += 1,
                }
            }
            assert_simd_matches_scalar(&noisy_frame(sw, sh, next_rand(&mut state)), sw, sh, dw, dh);
        }
        if !cfg!(miri) {
            assert!(narrow > 100 && wide > 0, "both rounding paths exercised: {narrow} narrow, {wide} wide");
        }
        // Every tail length after whole 8-output blocks, at 2, 3 and 4 taps.
        for dw in 16..=24u32 {
            for sw in [dw * 3 / 2 + 1, dw * 2 + 1, dw * 3 + 1] {
                assert_simd_matches_scalar(&noisy_frame(sw, 6, dw), sw, 6, dw, 4);
            }
        }
    }

    /// Saturated frames at the largest denominators the narrow (SIMD) path
    /// takes: the u32 sums and the 32-bit reciprocal must not overflow or
    /// round differently at the top of the range.
    #[test]
    fn scaler_simd_matches_scalar_on_saturated_frames() {
        // den = x.total * y.total: 64*64 = 4096 (the narrow limit), 63*65,
        // 4097 → wide, 2*2.
        for &((sw, sh), (dw, dh)) in &[((64u32, 64u32), (63u32, 63u32)), ((63, 65), (62, 64)), ((4097, 2), (4096, 1)), ((40, 8), (20, 4))] {
            let white = vec![255u8; i420_len(sw, sh)];
            assert_simd_matches_scalar(&white, sw, sh, dw, dh);
            assert!(scale_i420(&white, sw, sh, dw, dh).iter().all(|&p| p == 255), "{sw}x{sh} -> {dw}x{dh}");
            assert_simd_matches_scalar(&noisy_frame(sw, sh, 3), sw, sh, dw, dh);
        }
        let scaler = I420Scaler::new(64, 64, 63, 63);
        assert!(matches!(scaler.luma.rounding, Rounding::Narrow { .. }), "den 4096 takes the narrow path");
    }

    #[test]
    fn scale_keeps_a_constant_frame_constant() {
        for &((sw, sh), (dw, dh)) in &[((12u32, 8u32), (8u32, 6u32)), ((97, 55), (31, 17)), ((5, 5), (2, 3))] {
            let (y, c) = ((sw * sh) as usize, (sw.div_ceil(2) * sh.div_ceil(2)) as usize);
            let frame: Vec<u8> = [vec![77u8; y], vec![3u8; c], vec![250u8; c]].concat();
            let out = scale_i420(&frame, sw, sh, dw, dh);
            let (dy, dc) = ((dw * dh) as usize, (dw.div_ceil(2) * dh.div_ceil(2)) as usize);
            assert_eq!(out.len(), dy + 2 * dc, "{sw}x{sh} -> {dw}x{dh}");
            assert!(out[..dy].iter().all(|&p| p == 77), "{sw}x{sh} -> {dw}x{dh}: Y");
            assert!(out[dy..dy + dc].iter().all(|&p| p == 3), "{sw}x{sh} -> {dw}x{dh}: U");
            assert!(out[dy + dc..].iter().all(|&p| p == 250), "{sw}x{sh} -> {dw}x{dh}: V");
        }
    }

    /// An averaging filter with non-negative weights keeps a ramp a ramp.
    #[test]
    fn scale_keeps_a_gradient_monotone() {
        let (sw, sh, dw, dh) = (60u32, 40u32, 37u32, 23u32);
        let (scw, sch) = (sw.div_ceil(2), sh.div_ceil(2));
        let mut frame = Vec::new();
        for y in 0..sh {
            frame.extend((0..sw).map(|x| (x * 3 + y) as u8)); // rises left→right and top→bottom (max 216)
        }
        for _ in 0..2 {
            for y in 0..sch {
                frame.extend((0..scw).map(|x| (200 - x * 5 - y) as u8)); // falls
            }
        }
        let out = scale_i420(&frame, sw, sh, dw, dh);
        let (dw, dh) = (dw as usize, dh as usize);
        let y = &out[..dw * dh];
        for r in 0..dh {
            let row = &y[r * dw..(r + 1) * dw];
            assert!(row.windows(2).all(|p| p[0] <= p[1]), "Y row {r} not rising: {row:?}");
        }
        for c in 0..dw {
            assert!((1..dh).all(|r| y[(r - 1) * dw + c] <= y[r * dw + c]), "Y column {c} not rising");
        }
        let (dcw, dch) = (dw.div_ceil(2), dh.div_ceil(2));
        let u = &out[dw * dh..dw * dh + dcw * dch];
        for r in 0..dch {
            let row = &u[r * dcw..(r + 1) * dcw];
            assert!(row.windows(2).all(|p| p[0] >= p[1]), "U row {r} not falling: {row:?}");
        }
    }

    /// Output is tight I420 at the target: chroma planes round up, odd or even.
    #[test]
    fn scale_output_has_the_target_layout() {
        for &((sw, sh), (dw, dh)) in &[((9u32, 7u32), (5u32, 3u32)), ((8, 8), (6, 4)), ((8, 6), (7, 5)), ((3, 3), (1, 1))] {
            let (sy, sc) = ((sw * sh) as usize, (sw.div_ceil(2) * sh.div_ceil(2)) as usize);
            let frame: Vec<u8> = [vec![10u8; sy], vec![20u8; sc], vec![30u8; sc]].concat();
            let out = scale_i420(&frame, sw, sh, dw, dh);
            let (dy, dc) = ((dw * dh) as usize, (dw.div_ceil(2) * dh.div_ceil(2)) as usize);
            assert_eq!(out.len(), i420_len(dw, dh));
            let expected: Vec<u8> = [vec![10u8; dy], vec![20u8; dc], vec![30u8; dc]].concat();
            assert_eq!(out, expected, "{sw}x{sh} -> {dw}x{dh}");
        }
    }

    /// The reciprocal multiply rounds exactly like `(sum + den/2) / den`:
    /// exhaustively for small denominators, and around every rounding
    /// boundary for large ones, including both sides of the 2^28 switch to
    /// plain division.
    #[test]
    fn round_div_is_exact() {
        let plain = |sum: u64, den: u64| ((sum + den / 2) / den) as u8;
        // Pure integer code, nothing for Miri to check: keep it cheap there.
        let exhaustive = if cfg!(miri) { 12 } else { 200u64 };
        for den in 1..=exhaustive {
            let d = RoundDiv::new(den);
            for sum in 0..=255 * den {
                assert_eq!(d.div(sum), plain(sum, den), "{sum}/{den}");
            }
        }
        for den in [9u64, 1_234_567, 3839 * 2159, (1 << 28) - 1, 1 << 28, (1 << 28) + 1, 4095 * 4093, u64::from(u32::MAX)] {
            let d = RoundDiv::new(den);
            for k in 0..=255u64 {
                // Just below, at and above the half-way point between k and k+1.
                let mid = k * den + den / 2;
                for sum in [k * den, mid.saturating_sub(1), mid, mid + 1, (k + 1) * den - 1] {
                    if sum <= 255 * den {
                        assert_eq!(d.div(sum), plain(sum, den), "{sum}/{den}");
                    }
                }
            }
        }
    }

    /// Throughput at real sizes: [`I420Scaler::scale_into`] reusing the
    /// scaler and the output buffer (what the GOP adapter and the push
    /// sessions do per frame), and one-shot [`scale_i420`] (tables + a
    /// fresh output each call). Not a gate: run it by hand in release —
    /// `cargo test --release --features video --lib video_frame::tests::scale_benchmark -- --ignored --nocapture`
    /// (add `--target x86_64-apple-darwin` for x86_64; under Rosetta that has
    /// no AVX2, so it times the scalar fallback).
    #[test]
    #[ignore = "benchmark; run by hand in release"]
    fn scale_benchmark() {
        let cases = [
            ((1920u32, 1080u32), (1280u32, 720u32)),
            ((1920, 1080), (848, 480)),
            ((3840, 2160), (1920, 1080)),
            ((3840, 2160), (1280, 720)),
        ];
        let best_ms = |f: &mut dyn FnMut()| {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let start = std::time::Instant::now();
                for _ in 0..10 {
                    f();
                }
                best = best.min(start.elapsed().as_secs_f64() * 1e3 / 10.0);
            }
            best
        };
        eprintln!("kernel: {:?}", Isa::detect());
        for ((sw, sh), (dw, dh)) in cases {
            let frame = noisy_frame(sw, sh, 7);
            let mut scaler = I420Scaler::new(sw, sh, dw, dh);
            let mut out = Vec::new();
            let streamed = best_ms(&mut || {
                out.clear();
                scaler.scale_into(&frame, &mut out);
            });
            let mut check = 0u64;
            let one_shot = best_ms(&mut || check += u64::from(scale_i420(&frame, sw, sh, dw, dh)[1234]));
            eprintln!(
                "{sw}x{sh} -> {dw}x{dh}: {streamed:.2} ms/frame reused, {one_shot:.2} ms one-shot (check {})",
                out[1234]
            );
            let _ = check;
        }
    }

    #[test]
    #[should_panic(expected = "upscale")]
    fn scale_refuses_to_upscale() {
        let f = numbered_frame(6, 4, 0);
        let _ = scale_i420(&f, 6, 4, 8, 4);
    }

    #[test]
    #[should_panic(expected = "not a 6x4 I420 frame")]
    fn scale_rejects_a_wrong_sized_frame() {
        let _ = scale_i420(&[0u8; 10], 6, 4, 4, 2);
    }

    #[test]
    fn frames_scale_one_by_one() {
        let (w, h) = (12u32, 8u32);
        let a = noisy_frame(w, h, 1);
        let b = noisy_frame(w, h, 2);
        let both = [a.clone(), b.clone()].concat();
        let expected = [scale_i420(&a, w, h, 8, 6), scale_i420(&b, w, h, 8, 6)].concat();
        assert_eq!(scale_i420_frames(&both, w, h, 8, 6), expected);
        assert!(scale_i420_frames(&[], w, h, 8, 6).is_empty());
    }

    // ─── OrientedGopSource with scaling ──────────────────────────

    /// Two GOPs of 2 frames plus a trailing 1-frame GOP, 12x8.
    #[cfg(any(feature = "h264-encoder", all(feature = "av1-encoder", feature = "av1-decoder")))]
    fn scalable_clip() -> (Vec<u8>, u32, u32, u32, u32) {
        let (w, h, n, gop) = (12u32, 8u32, 5u32, 2u32);
        let clip = (0..n).flat_map(|f| noisy_frame(w, h, f)).collect();
        (clip, w, h, n, gop)
    }

    /// Scale in the coded orientation first, then turn: a 12x8 coded clip
    /// scaled to 8x6 and turned a quarter comes out 6x8.
    #[cfg(feature = "h264-encoder")]
    #[test]
    fn scaled_h264_source_scales_then_turns_each_gop() {
        use crate::codec::h264::openh264_stego::{GopYuvSource, SliceYuvSource};
        let (clip, w, h, n, gop) = scalable_clip();
        let mut plain = SliceYuvSource::new(&clip, w, h, n, gop);
        let mut src = OrientedGopSource::new_scaled(SliceYuvSource::new(&clip, w, h, n, gop), w, h, Some((8, 6)), 1);
        assert_eq!(src.output_dims(), (6, 8));
        src.rewind().unwrap();
        plain.rewind().unwrap();
        for g in 0..3 {
            let expected = orient_i420_frames(&scale_i420_frames(&plain.gop_yuv(g).unwrap(), w, h, 8, 6), 8, 6, 1);
            let got = src.gop_yuv(g).unwrap();
            assert_eq!(got.len(), if g < 2 { 2 } else { 1 } * i420_len(6, 8), "GOP {g}");
            assert_eq!(got, expected, "GOP {g}");
        }
        assert!(src.gop_yuv(3).unwrap().is_empty(), "end of stream passes through");
    }

    #[cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]
    #[test]
    fn scaled_av1_source_scales_mirrors_and_turns_and_rewinds() {
        use crate::codec::av1::stego::whole_video::{Av1GopYuvSource, Av1SliceYuvSource};
        let (clip, w, h, n, gop) = scalable_clip();
        let mut plain = Av1SliceYuvSource::new(&clip, w, h, n, gop);
        let code = ORIENT_MIRROR | 3;
        let mut src =
            OrientedGopSource::new_scaled(Av1SliceYuvSource::new(&clip, w, h, n, gop), w, h, Some((8, 4)), code);
        assert_eq!(src.output_dims(), (4, 8));
        assert!(src.gop_yuv(0).is_err(), "gop_yuv before rewind must fail as the inner source does");
        src.rewind().unwrap();
        plain.rewind().unwrap();
        for g in 0..3 {
            let expected = orient_i420_frames(&scale_i420_frames(&plain.gop_yuv(g).unwrap(), w, h, 8, 4), 8, 4, code);
            assert_eq!(src.gop_yuv(g).unwrap(), expected, "GOP {g}");
        }
        assert!(src.gop_yuv(3).unwrap().is_empty());
    }

    /// No scale is exactly the unscaled source, and so is a scale to the
    /// coded size itself.
    #[cfg(feature = "h264-encoder")]
    #[test]
    fn scaled_source_without_a_scale_is_the_oriented_source() {
        use crate::codec::h264::openh264_stego::{GopYuvSource, SliceYuvSource};
        let (clip, w, h, n, gop) = scalable_clip();
        for scale in [None, Some((w, h))] {
            for code in [0, 1, ORIENT_MIRROR | 2] {
                let mut oriented = OrientedGopSource::new(SliceYuvSource::new(&clip, w, h, n, gop), w, h, code);
                let mut scaled = OrientedGopSource::new_scaled(SliceYuvSource::new(&clip, w, h, n, gop), w, h, scale, code);
                assert_eq!(scaled.output_dims(), oriented.output_dims());
                for g in 0..4 {
                    assert_eq!(scaled.gop_yuv(g).unwrap(), oriented.gop_yuv(g).unwrap(), "{scale:?} {code} GOP {g}");
                }
            }
        }
    }

    /// A GOP that is not whole coded frames is an error, not a panic (a
    /// panic aborts the mobile apps).
    #[cfg(feature = "h264-encoder")]
    #[test]
    fn scaled_source_reports_a_ragged_gop_as_an_error() {
        use crate::codec::h264::openh264_stego::GopYuvSource;
        let ragged = vec![0u8; i420_len(12, 8) + 5];
        let mut scaled = OrientedGopSource::new_scaled(FixedGop(ragged.clone()), 12, 8, Some((8, 6)), 0);
        assert!(scaled.gop_yuv(0).is_err(), "scaled");
        let mut turned = OrientedGopSource::new(FixedGop(ragged), 12, 8, 1);
        assert!(turned.gop_yuv(0).is_err(), "turned");
    }

    /// A source that hands back the same buffer for every GOP.
    #[cfg(feature = "h264-encoder")]
    struct FixedGop(Vec<u8>);

    #[cfg(feature = "h264-encoder")]
    impl crate::codec::h264::openh264_stego::GopYuvSource for FixedGop {
        fn gop_yuv(&mut self, _gop_index: u32) -> Result<Vec<u8>, crate::stego::error::StegoError> {
            Ok(self.0.clone())
        }
        fn rewind(&mut self) -> Result<(), crate::stego::error::StegoError> {
            Ok(())
        }
    }

    #[test]
    #[should_panic(expected = "upscale")]
    fn scaled_source_refuses_an_upscale() {
        let _ = OrientedGopSource::new_scaled((), 12, 8, Some((14, 8)), 0);
    }
}
