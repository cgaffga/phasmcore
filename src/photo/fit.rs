// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Fit a photo into what is left of a message's budget (primary or shadow).
//!
//! The rule measured on 20 photos (2026-11-07, `docs/design/app/photo-attachments.md`): go down the
//! resolution ladder from 1280 px and take the largest size whose best fitting quality is at or above
//! the floor (libaom quantizer 58, qindex 232); if none is, the smallest size at whatever quality
//! fits. Search with libaom speed 6, then encode the chosen size at speed 2 from the speed-6
//! quality, stepping the quantizer until the budget is just filled. The budget is checked on the
//! whole payload ([`compressed_payload_size`] of the message's text, its other files and this photo),
//! so it is exact.

use super::avif::{write_avif, NCLX_SRGB_BT601_FULL};
use super::yuv::rgb_to_i420;
use super::PhotoError;
use crate::codec::resample::{fit_long_side, resample_rgb};
use crate::stego::payload::{compressed_payload_size, FileEntry};

/// Long sides tried, largest first (the measurement's ladder from the 1280 px cap down).
pub const LADDER: [u32; 11] = [1280, 1024, 800, 640, 512, 400, 320, 256, 200, 160, 128];
/// The quality floor: libaom quantizer 58 = qindex 232 (maintainer, 2026-11-08).
pub const FLOOR_QUANTIZER: i32 = 58;
const SEARCH_SPEED: i32 = 6;
const FINAL_SPEED: i32 = 2;
/// The source is first brought down to this long side (twice the cap): every ladder size is then
/// resampled from it, not from the full camera image each time.
const WORKING_LONG_SIDE: u32 = 2560;
/// Relative time of one speed-2 encode against a speed-6 one of the same size (measured 12-16x).
const FINAL_COST: u64 = 14;

/// A fitted photo.
#[derive(Debug, Clone)]
pub struct FittedPhoto {
    /// The AVIF file (the attachment's content).
    pub avif: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// libaom quantizer of the final encode (0-63, lower is better).
    pub quantizer: u8,
    /// The quality is below the floor: even the smallest size only fits at a lower quality.
    pub below_floor: bool,
}

/// Even `(w, h)` with long side `long` (the measurement's `dims_for`).
fn dims_for(w: u32, h: u32, long: u32) -> (u32, u32) {
    let even = |v: f64| ((v / 2.0).round() as u32 * 2).max(2);
    if w >= h {
        (long, even(h as f64 * long as f64 / w as f64))
    } else {
        (even(w as f64 * long as f64 / h as f64), long)
    }
}

struct Fitter<'a> {
    rgb: Vec<u8>,
    w: u32,
    h: u32,
    text: &'a str,
    files: &'a [FileEntry],
    filename: &'a str,
    target: usize,
    threads: i32,
    planes: Vec<(u32, (u32, u32, Vec<u8>, Vec<u8>, Vec<u8>))>,
    progress: &'a mut dyn FnMut(f32) -> bool,
    done: u64,
    shown: f32,
}

impl Fitter<'_> {
    fn planes(&mut self, long: u32) -> Result<usize, PhotoError> {
        if let Some(i) = self.planes.iter().position(|(l, _)| *l == long) {
            return Ok(i);
        }
        let (dw, dh) = dims_for(self.w, self.h, long);
        let px = resample_rgb(&self.rgb, self.w, self.h, dw, dh).map_err(|_| PhotoError::InvalidImage)?;
        let (y, u, v) = rgb_to_i420(&px, dw as usize, dh as usize);
        self.planes.push((long, (dw, dh, y, u, v)));
        Ok(self.planes.len() - 1)
    }

    /// The AVIF at this size, quantizer and speed, and whether the payload with it fits.
    fn encode(&mut self, long: u32, q: i32, speed: i32, remaining: u64) -> Result<(Vec<u8>, u32, u32, bool), PhotoError> {
        let i = self.planes(long)?;
        let (w, h) = (self.planes[i].1 .0, self.planes[i].1 .1);
        let file = {
            let (_, (_, _, y, u, v)) = &self.planes[i];
            encode_i420(y, u, v, w, h, q, speed, self.threads)?
        };
        let mut all = self.files.to_vec();
        all.push(FileEntry { filename: self.filename.to_string(), content: file });
        let fits = compressed_payload_size(self.text, &all) <= self.target;
        let file = all.pop().unwrap().content;

        self.done += w as u64 * h as u64 * if speed == FINAL_SPEED { FINAL_COST } else { 1 };
        let p = (self.done as f64 / (self.done + remaining) as f64) as f32;
        self.shown = self.shown.max(p.min(0.99));
        if !(self.progress)(self.shown) {
            return Err(PhotoError::Cancelled);
        }
        Ok((file, w, h, fits))
    }

    /// Estimated cost of the final stage at this size.
    fn final_cost(&self, long: u32) -> u64 {
        let (w, h) = dims_for(self.w, self.h, long);
        2 * FINAL_COST * w as u64 * h as u64
    }
}

/// One AVIF file from tight 8-bit 4:2:0 planes (BT.601 full range; `w`, `h` even) at libaom
/// quantizer `q` (0-63) and `speed`.
#[allow(clippy::too_many_arguments)]
pub fn encode_i420(y: &[u8], u: &[u8], v: &[u8], w: u32, h: u32, q: i32, speed: i32, threads: i32) -> Result<Vec<u8>, PhotoError> {
    let (wu, hu) = (w as usize, h as usize);
    if w < 2 || h < 2 || w % 2 == 1 || h % 2 == 1 || y.len() < wu * hu || u.len() < wu * hu / 4 || v.len() < wu * hu / 4 {
        return Err(PhotoError::InvalidImage);
    }
    let mut out: *mut u8 = core::ptr::null_mut();
    let mut len = 0usize;
    // SAFETY: the planes hold w x h and w/2 x h/2 bytes (checked above); the shim writes a
    // malloc'd buffer to `out`, freed below with phasm_aom_free.
    let rc = unsafe {
        core_aom_sys::phasm_aom_encode_i420(y.as_ptr(), u.as_ptr(), v.as_ptr(), w, h, q, speed, threads, &mut out, &mut len)
    };
    if rc != 0 || out.is_null() {
        return Err(PhotoError::Encoder(rc));
    }
    // SAFETY: `out` holds `len` bytes written by the shim.
    let av1 = unsafe { core::slice::from_raw_parts(out, len) }.to_vec();
    unsafe { core_aom_sys::phasm_aom_free(out) };
    write_avif(&av1, w, h, NCLX_SRGB_BT601_FULL)
}

/// Fit interleaved 8-bit RGB `rgb` (`width` x `height`, upright sRGB) as the attachment `filename`
/// of a message with `text` and `files` (the message's other attachments), so that the message's
/// whole payload stays within `target_payload_bytes` (the size the capacity bar compares, e.g. the
/// top of the yellow band). `progress` gets 0..1 and returns false to cancel.
pub fn fit_photo(
    rgb: &[u8],
    width: u32,
    height: u32,
    text: &str,
    files: &[FileEntry],
    filename: &str,
    target_payload_bytes: usize,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Result<FittedPhoto, PhotoError> {
    let need = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(3));
    if width == 0 || height == 0 || need.is_none_or(|n| rgb.len() < n) {
        return Err(PhotoError::InvalidImage);
    }
    if !progress(0.0) {
        return Err(PhotoError::Cancelled);
    }
    // the working copy: at most WORKING_LONG_SIDE
    let (ww, wh) = fit_long_side(width, height, WORKING_LONG_SIDE);
    let work = resample_rgb(rgb, width, height, ww, wh).map_err(|_| PhotoError::InvalidImage)?;
    let threads = std::thread::available_parallelism().map(|n| n.get().min(4)).unwrap_or(2) as i32;
    let mut f = Fitter {
        rgb: work,
        w: ww,
        h: wh,
        text,
        files,
        filename,
        target: target_payload_bytes,
        threads,
        planes: Vec::new(),
        progress,
        done: 0,
        shown: 0.0,
    };

    // the ladder for this photo: never enlarged; a photo smaller than 128 px keeps its size
    let long = ww.max(wh);
    let mut ladder: Vec<u32> = LADDER.iter().copied().filter(|&l| l <= long).collect();
    if ladder.is_empty() {
        ladder.push((long / 2 * 2).max(2));
    }
    let n = ladder.len();

    // 1. the largest size that fits at the floor (binary search, smallest size first)
    let smallest = ladder[n - 1];
    let (_, _, _, ok) = f.encode(smallest, FLOOR_QUANTIZER, SEARCH_SPEED, f.final_cost(smallest) + 8 * 400 * 400)?;
    let (long, q6, _) = if ok {
        let (mut lo, mut hi) = (0usize, n - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let est = f.final_cost(ladder[mid]) + 6 * 400 * 400;
            if f.encode(ladder[mid], FLOOR_QUANTIZER, SEARCH_SPEED, est)?.3 {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let l = ladder[hi];
        // 2. the best quantizer at that size that still fits (58 fits): binary search 0..=58
        let (mut good, mut bad) = (FLOOR_QUANTIZER, -1);
        while good - bad > 1 {
            let mid = (good + bad) / 2;
            if f.encode(l, mid, SEARCH_SPEED, f.final_cost(l))?.3 {
                good = mid;
            } else {
                bad = mid;
            }
        }
        (l, good, false)
    } else {
        // nothing fits at the floor: the smallest size at the best quality that fits
        let mut q = None;
        for cand in FLOOR_QUANTIZER + 1..=63 {
            if f.encode(smallest, cand, SEARCH_SPEED, f.final_cost(smallest))?.3 {
                q = Some(cand);
                break;
            }
        }
        (smallest, q.ok_or(PhotoError::DoesNotFit)?, true)
    };

    // 3. the final encode at speed 2, from the speed-6 quantizer, filling the budget
    let mut best: Option<(Vec<u8>, u32, u32, i32)> = None;
    let (file, w, h, fits) = f.encode(long, q6, FINAL_SPEED, f.final_cost(long) / 2)?;
    if fits {
        best = Some((file, w, h, q6));
        // a few steps (speed 2 is a few percent smaller than speed 6); at the size cap quality is
        // the only way to use the budget, so more there
        let steps = if long == ladder[0] && long == LADDER[0] { 12 } else { 4 };
        let mut q = q6 - 1;
        while q >= 0 && q >= q6 - steps {
            let (file, w, h, fits) = f.encode(long, q, FINAL_SPEED, f.final_cost(long) / 2)?;
            if !fits {
                break;
            }
            best = Some((file, w, h, q));
            q -= 1;
        }
    } else {
        for q in q6 + 1..=63 {
            let (file, w, h, fits) = f.encode(long, q, FINAL_SPEED, f.final_cost(long) / 2)?;
            if fits {
                best = Some((file, w, h, q));
                break;
            }
        }
    }
    let (avif, w, h, q) = match best {
        Some(b) => b,
        // speed 2 never fitted (not seen in the measurement): the speed-6 result fits
        None => {
            let (file, w, h, fits) = f.encode(long, q6, SEARCH_SPEED, 0)?;
            if !fits {
                return Err(PhotoError::DoesNotFit);
            }
            (file, w, h, q6)
        }
    };
    (f.progress)(1.0);
    Ok(FittedPhoto { avif, width: w, height: h, quantizer: q as u8, below_floor: q > FLOOR_QUANTIZER })
}
