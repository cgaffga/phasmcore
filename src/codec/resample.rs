// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! The one image resampler every frontend uses for a cover (resolution chip,
//! Armor's channel sizes): separable Lanczos-3 on 8-bit sRGB, the kernel
//! widened by the scale on downscales.
//!
//! Before, iOS resampled with Core Graphics (and sometimes the lossy QF92
//! cover or the 1600 px preview), Android drew an unfiltered `drawBitmap`, and
//! the web used the canvas: three different covers for one photo, and an
//! aliased one on Android (fix list 2.3). Here the weights come from
//! `det_math`'s sine and are fixed point (16 bits, each tap set summing to
//! exactly 1.0), and both passes run in integers, so iOS, Android, WASM and
//! the CLI produce the same bytes. A resampled image is a valid Deep Cover
//! precover (`docs/design/image/deep-cover-exact-si.md`): the frontends hand
//! exactly these pixels to the cover writer and to `encode_si*`.

use crate::det_math::det_sin;

/// Fixed-point weight scale: each output sample's taps sum to `1 << WEIGHT_BITS`.
const WEIGHT_BITS: u32 = 16;

/// Lanczos-3 taps of one output coordinate: first source index and weights.
struct Taps {
    first: usize,
    weights: Vec<i32>,
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        det_sin(px) / px
    }
}

/// Taps for every destination coordinate of a `src` -> `dst` axis: sample
/// centres aligned (`(d + 0.5) · scale − 0.5`), support 3 · max(scale, 1),
/// taps outside the image dropped and the rest renormalized, then rounded to
/// fixed point with the rounding residue on the largest tap.
fn axis_taps(src: usize, dst: usize) -> Vec<Taps> {
    let scale = src as f64 / dst as f64;
    let support = 3.0 * scale.max(1.0);
    let inv = 1.0 / scale.max(1.0);
    let one = 1i32 << WEIGHT_BITS;
    (0..dst)
        .map(|d| {
            let c = (d as f64 + 0.5) * scale - 0.5;
            let lo = (c - support).ceil().max(0.0) as usize;
            let hi = ((c + support).floor().max(0.0) as usize).min(src - 1);
            let w: Vec<f64> = (lo..=hi)
                .map(|i| {
                    let x = (i as f64 - c) * inv;
                    if x.abs() >= 3.0 { 0.0 } else { sinc(x) * sinc(x / 3.0) }
                })
                .collect();
            let sum: f64 = w.iter().sum();
            let mut weights: Vec<i32> = w.iter().map(|v| (v / sum * one as f64).round() as i32).collect();
            let residue = one - weights.iter().sum::<i32>();
            let big = (0..weights.len()).max_by_key(|&i| weights[i]).unwrap_or(0);
            weights[big] += residue;
            Taps { first: lo, weights }
        })
        .collect()
}

#[inline]
fn round_clamp(acc: i64) -> u8 {
    ((acc + (1 << (WEIGHT_BITS - 1))) >> WEIGHT_BITS).clamp(0, 255) as u8
}

/// The largest `(w, h)` with the same aspect ratio whose long side is at most
/// `max_long` (unchanged when the image already fits; sides at least 1).
pub fn fit_long_side(width: u32, height: u32, max_long: u32) -> (u32, u32) {
    let long = width.max(height);
    if long <= max_long || long == 0 {
        return (width, height);
    }
    let scale = |side: u32| ((side as u64 * max_long as u64 + long as u64 / 2) / long as u64).max(1) as u32;
    if width >= height { (max_long, scale(height)) } else { (scale(width), max_long) }
}

/// Resample interleaved 8-bit RGB `pixels` (`width × height`) to
/// `dst_width × dst_height`. Deterministic on every target.
///
/// # Errors
/// [`crate::StegoError::ImageTooSmall`] if a size is 0 or the buffer is
/// shorter than `width · height · 3`.
pub fn resample_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Result<Vec<u8>, crate::StegoError> {
    let (sw, sh, dw, dh) = (width as usize, height as usize, dst_width as usize, dst_height as usize);
    let need = sw.checked_mul(sh).and_then(|n| n.checked_mul(3)).unwrap_or(usize::MAX);
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 || pixels.len() < need {
        return Err(crate::StegoError::ImageTooSmall);
    }
    if (sw, sh) == (dw, dh) {
        return Ok(pixels[..need].to_vec());
    }
    // Horizontal pass into an 8-bit intermediate (dw × sh), then vertical.
    let xt = axis_taps(sw, dw);
    let mut mid = vec![0u8; dw * sh * 3];
    let row_h = |y: usize, out: &mut [u8]| {
        let src = &pixels[y * sw * 3..(y + 1) * sw * 3];
        for (x, t) in xt.iter().enumerate() {
            let mut acc = [0i64; 3];
            for (k, &wt) in t.weights.iter().enumerate() {
                let p = (t.first + k) * 3;
                for c in 0..3 {
                    acc[c] += wt as i64 * src[p + c] as i64;
                }
            }
            for c in 0..3 {
                out[x * 3 + c] = round_clamp(acc[c]);
            }
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        mid.par_chunks_mut(dw * 3).enumerate().for_each(|(y, out)| row_h(y, out));
    }
    #[cfg(not(feature = "parallel"))]
    for (y, out) in mid.chunks_mut(dw * 3).enumerate() {
        row_h(y, out);
    }

    let yt = axis_taps(sh, dh);
    let mut dst = vec![0u8; dw * dh * 3];
    let row_v = |y: usize, out: &mut [u8]| {
        let t = &yt[y];
        let mut acc = vec![0i64; dw * 3];
        for (k, &wt) in t.weights.iter().enumerate() {
            let src = &mid[(t.first + k) * dw * 3..(t.first + k + 1) * dw * 3];
            for (a, &s) in acc.iter_mut().zip(src) {
                *a += wt as i64 * s as i64;
            }
        }
        for (o, a) in out.iter_mut().zip(acc) {
            *o = round_clamp(a);
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        dst.par_chunks_mut(dw * 3).enumerate().for_each(|(y, out)| row_v(y, out));
    }
    #[cfg(not(feature = "parallel"))]
    for (y, out) in dst.chunks_mut(dw * 3).enumerate() {
        row_v(y, out);
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textured(w: usize, h: usize) -> Vec<u8> {
        let mut px = Vec::with_capacity(w * h * 3);
        let mut s: u32 = 7;
        for y in 0..h {
            for x in 0..w {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let n = (s >> 27) as u8;
                px.push(((x * 3 + y) as u8).wrapping_add(n));
                px.push(((x / 2 + y * 2) as u8) / 2 + n);
                px.push((((x ^ y) & 0xFF) as u8) / 2);
            }
        }
        px
    }

    #[test]
    fn taps_sum_to_one_and_identity_is_exact() {
        for (s, d) in [(4032, 1600), (1600, 1280), (1280, 1080), (100, 100), (7, 3), (3, 7)] {
            for t in axis_taps(s, d) {
                assert_eq!(t.weights.iter().sum::<i32>(), 1 << WEIGHT_BITS, "{s}->{d}");
            }
        }
        let px = textured(37, 21);
        assert_eq!(resample_rgb(&px, 37, 21, 37, 21).unwrap(), px);
        // Same size through the taps (not the shortcut): sinc at integers.
        for t in axis_taps(50, 50) {
            assert_eq!(t.weights.iter().filter(|&&w| w != 0).count(), 1);
        }
    }

    #[test]
    fn flat_stays_flat_and_sizes_are_right() {
        let px = vec![123u8; 640 * 480 * 3];
        let out = resample_rgb(&px, 640, 480, 333, 250).unwrap();
        assert_eq!(out.len(), 333 * 250 * 3);
        assert!(out.iter().all(|&v| v == 123));
        assert_eq!(fit_long_side(4032, 3024, 1600), (1600, 1200));
        assert_eq!(fit_long_side(3024, 4032, 1280), (960, 1280));
        assert_eq!(fit_long_side(1000, 800, 1600), (1000, 800));
        assert!(resample_rgb(&px, 640, 480, 0, 10).is_err());
        assert!(resample_rgb(&px[..10], 640, 480, 10, 10).is_err());
    }

    /// Close to an independent Lanczos-3 (the `image` crate's) on a
    /// downscale: same filter family, different rounding.
    #[test]
    fn matches_a_reference_lanczos() {
        let (w, h) = (403usize, 301usize);
        let px = textured(w, h);
        let (dw, dh) = (161u32, 120u32);
        let ours = resample_rgb(&px, w as u32, h as u32, dw, dh).unwrap();
        let img = image::RgbImage::from_raw(w as u32, h as u32, px).unwrap();
        let theirs = image::imageops::resize(&img, dw, dh, image::imageops::FilterType::Lanczos3).into_raw();
        let mse: f64 = ours.iter().zip(&theirs).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum::<f64>() / ours.len() as f64;
        let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
        assert!(psnr > 40.0, "PSNR vs reference {psnr:.1} dB");
    }
}
