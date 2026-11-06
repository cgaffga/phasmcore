// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Armor cover dimensions: the one size rule every frontend follows.
//!
//! A messenger that downscales an Armor image (a 1600 px cover sent through
//! Telegram arrives at 1280 px) loses the 8x8 block grid; the decoder's
//! rescale attempt (`pipeline::try_rescaled_decode`) scales the image back to
//! the cover size. It knows the cover's long side (the preset) but must guess
//! the short side from the received aspect ratio, and the messenger rounded
//! that side (1600x1067 -> 1280x854; back: 1067.5). A one-pixel error costs
//! STDM survival.
//!
//! The rule: the free side of every Armor cover (the side the preset does not
//! pin) is a multiple of [`ARMOR_COVER_STEP`] = 5, trimmed by a center crop
//! (at most 4 px, 0.45 % of the short side). Then:
//! - the most common downscale, 1600 -> 1280 px (x 4/5: Telegram, WhatsApp
//!   at 1280), lands on whole pixels: the received size is exact whatever the
//!   messenger's rounding, and so is the guess. Even a resampler that maps
//!   both axes with the long-side ratio (instead of per axis) then matches the
//!   decoder's inverse exactly;
//! - for other ratios (1600 -> 1080 wide: x 27/40; 1280 -> 1080: x 27/32)
//!   the decoder snaps its guess to the nearest multiple of 5 when it lies
//!   strictly within one *received* pixel of it ([`SNAP_TOLERANCE`]; s =
//!   cover px per received px). Whatever the messenger did (round, floor or
//!   ceil), the received side is within one pixel of the exact product, so
//!   the cover size is within s of the guess; and with s < 2.5 (downscales
//!   to more than 0.4) no other multiple of 5 is. The snap is then exact in
//!   every rounding mode.
//! - untrimmed covers (made before this rule) keep today's guess at x 4/5:
//!   there the guess is a multiple of s = 1.25 px, so a multiple of 5 strictly
//!   within 1.25 px of it is the guess itself. At other ratios a guess moves
//!   only when a multiple of 5 is within one received pixel.
//!
//! No extra decode attempt. Step 8 (block-aligned), step 16 and a full snap
//! were measured too: they lose the exact x 4/5 (which also serves resamplers
//! that map both axes with the long-side ratio) and cost untrimmed Fortress
//! covers.
//! See `eval/runs/2026-10-31-armor-v2/RESULTS.md`, Part 4.

/// The free side of an Armor cover is a multiple of this (pixels).
pub const ARMOR_COVER_STEP: u32 = 5;

/// The decoder snaps its cover-size guess to a multiple of
/// [`ARMOR_COVER_STEP`] only strictly within this distance, in received
/// pixels (one received pixel = cover long side / received long side).
pub const SNAP_TOLERANCE: f64 = 1.0;

/// Where the Armor message will be sent: fixes the cover's size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ArmorCoverPreset {
    /// Long side at most 1600 px (`ARMOR_TARGET_DIMENSION`): WhatsApp,
    /// Signal, e-mail. The default.
    Standard = 0,
    /// Long side at most 1280 px: Telegram (standard quality).
    Telegram = 1,
    /// Width at most 1080 px, aspect ratio within Instagram's 4:5 (portrait)
    /// to 1.91:1 (wide), center-cropped into that range so Instagram never
    /// crops the cover (any crop after encoding breaks Armor).
    Instagram = 2,
}

impl ArmorCoverPreset {
    /// The bridge id (0 = Standard, 1 = Telegram, 2 = Instagram).
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::Standard),
            1 => Some(Self::Telegram),
            2 => Some(Self::Instagram),
            _ => None,
        }
    }
}

/// The Armor cover for a source image: resize the upright source (EXIF
/// orientation applied) to `scaled_width` x `scaled_height` (with core's
/// `resample_rgb`, so every platform writes the same pixels), then take the
/// `width` x `height` rectangle at (`crop_x`, `crop_y`) of the resized image.
/// Encode that rectangle at QF65. The crop rectangle is what a HUD shows; in
/// source pixels it is the same rectangle times `src / scaled` per axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArmorCoverDims {
    pub width: u32,
    pub height: u32,
    pub scaled_width: u32,
    pub scaled_height: u32,
    pub crop_x: u32,
    pub crop_y: u32,
}

impl ArmorCoverDims {
    /// `[width, height, scaled_width, scaled_height, crop_x, crop_y]`, the
    /// order the bridges return.
    pub fn to_array(self) -> [u32; 6] {
        [self.width, self.height, self.scaled_width, self.scaled_height, self.crop_x, self.crop_y]
    }
}

/// Integer a * b / c rounded half up (identical on every platform).
fn mul_div_round(a: u32, b: u32, c: u32) -> u32 {
    let (a, b, c) = (a as u64, b as u64, c as u64);
    ((2 * a * b + c) / (2 * c)) as u32
}

/// The Armor cover size for a `src_width` x `src_height` source (upright,
/// EXIF orientation applied) and a channel preset. Every frontend calls this
/// so all platforms produce the same cover. No upscaling: a source within the
/// preset keeps its size (only the trim applies).
pub fn armor_cover_dimensions(src_width: u32, src_height: u32, preset: ArmorCoverPreset) -> ArmorCoverDims {
    armor_cover_dimensions_step(src_width, src_height, preset, ARMOR_COVER_STEP)
}

/// [`armor_cover_dimensions`] with another step (measurement; 1 = no trim).
#[doc(hidden)]
pub fn armor_cover_dimensions_step(src_width: u32, src_height: u32, preset: ArmorCoverPreset, step: u32) -> ArmorCoverDims {
    let (w, h) = (src_width, src_height);
    if w == 0 || h == 0 {
        return ArmorCoverDims { width: w, height: h, scaled_width: w, scaled_height: h, crop_x: 0, crop_y: 0 };
    }
    let (sw, sh) = match preset {
        ArmorCoverPreset::Standard | ArmorCoverPreset::Telegram => {
            let t = if preset == ArmorCoverPreset::Standard { crate::stego::ARMOR_TARGET_DIMENSION } else { 1280 };
            crate::codec::resample::fit_long_side(w, h, t)
        }
        ArmorCoverPreset::Instagram => return instagram(w, h, step.max(1)),
    };
    // The free side: the short side.
    let free_is_height = sw >= sh;
    let free = if free_is_height { sh } else { sw };
    let step = step.max(1);
    let kept = if free >= step { free / step * step } else { free };
    // Center crop; an odd trim leaves the extra pixel at the bottom / right.
    let off = (free - kept) / 2;
    if free_is_height {
        ArmorCoverDims { width: sw, height: kept, scaled_width: sw, scaled_height: sh, crop_x: 0, crop_y: off }
    } else {
        ArmorCoverDims { width: kept, height: sh, scaled_width: sw, scaled_height: sh, crop_x: off, crop_y: 0 }
    }
}

/// Instagram: width at most 1080 (pinned), aspect ratio within
/// [4:5, 1.91:1], the height a multiple of `step`, center crops.
fn instagram(w: u32, h: u32, step: u32) -> ArmorCoverDims {
    let floor_step = |v: u32| if v >= step { v / step * step } else { v };
    let mut cw = w.min(1080);
    let (mut sw, mut sh) = (cw, mul_div_round(h, cw, w).max(1));
    let mut ch = floor_step(sh);
    if 4 * ch as u64 > 5 * cw as u64 {
        // Taller than 4:5: crop the height (1080 x 1350 at full width).
        ch = floor_step(cw * 5 / 4);
    } else if 100 * cw as u64 > 191 * ch as u64 {
        // Wider than 1.91:1: the height is the smallest multiple of the step
        // at least 100 / 191 of the width; scale the source to that height
        // (a downscale) and crop the width.
        let need = floor_step((100 * cw).div_ceil(191) + step - 1);
        if need <= h {
            ch = need;
            (sw, sh) = (mul_div_round(w, need, h).max(cw), need);
        } else {
            // Too small to scale down to it (no upscaling): full height,
            // narrower width.
            (sw, sh) = (w, h);
            ch = floor_step(h);
            cw = (191 * ch as u64 / 100) as u32;
        }
    }
    ArmorCoverDims { width: cw, height: ch, scaled_width: sw, scaled_height: sh, crop_x: (sw - cw) / 2, crop_y: (sh - ch) / 2 }
}

/// Decoder side: the cover's short side from the received aspect ratio
/// (`guess` = received short x cover long / received long), snapped to the
/// nearest multiple of `step` when strictly within `tol` cover px of it,
/// else rounded; kept within [8, `long`].
pub(crate) fn snap_cover_short(guess: f64, long: usize, step: u32, tol: f64) -> usize {
    let k = step.max(1) as f64;
    let m = (guess / k).round() * k;
    let s = if (m - guess).abs() < tol { m } else { guess.round() };
    (s.max(8.0) as usize).min(long)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ArmorCoverPreset::*;

    fn dims(w: u32, h: u32, p: ArmorCoverPreset) -> (u32, u32, u32, u32, u32, u32) {
        let d = armor_cover_dimensions(w, h, p);
        (d.width, d.height, d.scaled_width, d.scaled_height, d.crop_x, d.crop_y)
    }

    /// `s`: cover px per received px (the decoder's tolerance unit).
    fn snap(guess: f64, long: usize, s: f64) -> usize {
        snap_cover_short(guess, long, ARMOR_COVER_STEP, SNAP_TOLERANCE * s)
    }

    #[test]
    fn presets() {
        // 3:2 landscape: 1600 x 1067 scaled, trimmed to 1065 (1 px top, 1 bottom).
        assert_eq!(dims(6000, 4000, Standard), (1600, 1065, 1600, 1067, 0, 1));
        // 4:3 and 16:9 land on the grid already.
        assert_eq!(dims(4032, 3024, Standard), (1600, 1200, 1600, 1200, 0, 0));
        assert_eq!(dims(3840, 2160, Standard), (1600, 900, 1600, 900, 0, 0));
        // Portrait: the width is free.
        assert_eq!(dims(4000, 6000, Standard), (1065, 1600, 1067, 1600, 1, 0));
        // Telegram.
        assert_eq!(dims(6000, 4000, Telegram), (1280, 850, 1280, 853, 0, 1));
        assert_eq!(dims(1600, 1067, Telegram), (1280, 850, 1280, 854, 0, 2));
        // Instagram: width 1080, height free (within 4:5 .. 1.91:1).
        assert_eq!(dims(6000, 4000, Instagram), (1080, 720, 1080, 720, 0, 0));
        assert_eq!(dims(4000, 5000, Instagram), (1080, 1350, 1080, 1350, 0, 0));
        assert_eq!(dims(1600, 1067, Instagram), (1080, 720, 1080, 720, 0, 0));
    }

    #[test]
    fn instagram_aspect_crop() {
        // 2:3 portrait: 1080 x 1620, cropped to 4:5 (1080 x 1350).
        assert_eq!(dims(4000, 6000, Instagram), (1080, 1350, 1080, 1620, 0, 135));
        assert_eq!(dims(3000, 4490, Instagram), (1080, 1350, 1080, 1616, 0, 133));
        // Small portrait: no upscale, 4:5 of its own width.
        assert_eq!(dims(800, 1201, Instagram), (800, 1000, 800, 1201, 0, 100));
        // 3:1 panorama: scaled to 570 px high (the smallest multiple of 5 at
        // least 1080 / 1.91), width cropped to 1080.
        assert_eq!(dims(3000, 1000, Instagram), (1080, 570, 1710, 570, 315, 0));
        // Small panorama: full height, width cropped to 1.91:1.
        assert_eq!(dims(900, 300, Instagram), (573, 300, 900, 300, 163, 0));
        // Just inside 1.91:1 before the trim: the 5 px trim tips it over.
        assert_eq!(dims(2160, 1132, Instagram), (1080, 570, 1088, 570, 4, 0));
        // In range: no crop beyond the trim; the other presets never crop to
        // an aspect ratio.
        assert_eq!(dims(1080, 1080, Instagram), (1080, 1080, 1080, 1080, 0, 0));
        assert_eq!(dims(3000, 1000, Standard), (1600, 530, 1600, 533, 0, 1));
        assert_eq!(dims(4000, 6000, Telegram), (850, 1280, 853, 1280, 1, 0));
    }

    #[test]
    fn no_upscale_and_small_sources() {
        // Within the preset: no scale, the free side still on the grid.
        assert_eq!(dims(1600, 1067, Standard), (1600, 1065, 1600, 1067, 0, 1));
        assert_eq!(dims(1000, 667, Standard), (1000, 665, 1000, 667, 0, 1));
        // Square: the height is the free side, already on the grid.
        assert_eq!(dims(2000, 2000, Standard), (1600, 1600, 1600, 1600, 0, 0));
        // Degenerate sizes do not panic.
        assert_eq!(dims(0, 10, Standard), (0, 10, 0, 10, 0, 0));
        assert_eq!(dims(20000, 5, Standard), (1600, 1, 1600, 1, 0, 0));
        assert_eq!(dims(7, 5, Standard), (7, 5, 7, 5, 0, 0));
        assert_eq!(dims(7, 4, Standard), (7, 4, 7, 4, 0, 0));
        assert_eq!(ArmorCoverPreset::from_id(3), None);
        assert_eq!(ArmorCoverPreset::from_id(1), Some(Telegram));
    }

    /// The rule's invariants over a sweep of source sizes and presets.
    #[test]
    fn invariants_sweep() {
        for p in [Standard, Telegram, Instagram] {
            for w in (200..5000).step_by(37) {
                for h in (200..5000).step_by(41) {
                    let d = armor_cover_dimensions(w, h, p);
                    // No upscaling, crop inside the scaled image.
                    assert!(d.scaled_width <= w && d.scaled_height <= h);
                    assert!(d.crop_x + d.width <= d.scaled_width && d.crop_y + d.height <= d.scaled_height);
                    if p == Instagram {
                        let (cw, ch) = (d.width as u64, d.height as u64);
                        assert!(cw <= 1080 && 4 * ch <= 5 * cw && 100 * cw <= 191 * ch, "{w}x{h} {d:?}");
                        assert_eq!(d.height % ARMOR_COVER_STEP, 0, "{w}x{h} {d:?}");
                        // Centered, and nothing cropped that the range allows.
                        assert!((d.scaled_width - d.width) / 2 == d.crop_x && (d.scaled_height - d.height) / 2 == d.crop_y);
                        assert!(d.width == d.scaled_width || d.scaled_height - d.height < ARMOR_COVER_STEP);
                        let sa = d.scaled_width as f64 / d.scaled_height as f64;
                        assert!((sa - w as f64 / h as f64).abs() < 0.01 * sa, "{w}x{h} {d:?}");
                        continue;
                    }
                    let free_h = d.scaled_width >= d.scaled_height;
                    let (free, scaled_free, off) =
                        if free_h { (d.height, d.scaled_height, d.crop_y) } else { (d.width, d.scaled_width, d.crop_x) };
                    assert_eq!(free % ARMOR_COVER_STEP, 0, "{w}x{h} {p:?} {d:?}");
                    assert!(scaled_free - free < ARMOR_COVER_STEP);
                    assert!(2 * off <= scaled_free - free && scaled_free - free - 2 * off <= 1);
                    // The pinned side is untouched.
                    if free_h {
                        assert_eq!(d.width, d.scaled_width);
                    } else {
                        assert_eq!(d.height, d.scaled_height);
                    }
                    let limit = if p == Standard { 1600 } else { 1280 };
                    assert!(d.scaled_width.max(d.scaled_height) <= limit);
                    // Aspect within half a pixel of the source.
                    let (sw, sh) = (d.scaled_width as f64, d.scaled_height as f64);
                    if free_h {
                        assert!((sh - h as f64 * sw / w as f64).abs() <= 0.5 + 1e-9);
                    } else {
                        assert!((sw - w as f64 * sh / h as f64).abs() <= 0.5 + 1e-9);
                    }
                }
            }
        }
    }

    /// Simulated messenger downscales of trimmed covers (x 4/5 and long sides
    /// 1280 / 1080 / 960 / 800, Instagram's width 1080): the snapped guess is
    /// the cover size in every rounding mode.
    #[test]
    fn snapped_guess_for_channel_downscales() {
        let modes: [(&str, fn(f64) -> f64); 3] = [("round", f64::round), ("floor", f64::floor), ("ceil", f64::ceil)];
        let (mut exact, mut checked) = (0usize, 0usize);
        for p in [Standard, Telegram] {
            for w in (1300..6000).step_by(7) {
                for h in [w * 2 / 3 + 1, w * 3 / 4 + 3, w * 9 / 16 + 1, w * 5 / 7, w * 10 / 11, w / 2 + 5] {
                    for (cw, ch) in [(w, h), (h, w)] {
                        let d = armor_cover_dimensions(cw, ch, p);
                        let (long, short) = (d.width.max(d.height) as f64, d.width.min(d.height) as f64);
                        // The 4/5 product is computed exactly as integers.
                        let four_fifths = short * 4.0 / 5.0;
                        for (name, m) in modes {
                            let g = m(four_fifths) * long / (long * 4.0 / 5.0);
                            assert_eq!(snap(g, long as usize, 1.25) as f64, short, "{cw}x{ch} {p:?} x4/5 {name}");
                            checked += 1;
                            exact += 1;
                        }
                        for target in [1280.0, 1080.0, 960.0, 800.0] {
                            if target >= long {
                                continue;
                            }
                            for (name, m) in modes {
                                let rx_short = m(short * target / long);
                                let got = snap(rx_short * long / target, long as usize, long / target) as f64;
                                checked += 1;
                                exact += (got == short) as usize;
                                assert_eq!(got, short, "{cw}x{ch} {p:?} -> {target} {name}");
                            }
                        }
                        // Instagram on a cover wider than 1080: width pinned.
                        if d.width > 1080 {
                            for (name, m) in modes {
                                let rx_h = m(d.height as f64 * 1080.0 / d.width as f64);
                                let (rl, rs) = (rx_h.max(1080.0), rx_h.min(1080.0));
                                let got = snap(rs * long / rl, long as usize, long / rl) as f64;
                                assert_eq!(got, short, "{cw}x{ch} {p:?} -> ig {name}");
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 10_000 && exact == checked);
    }

    /// Untrimmed covers (any short side) through x 4/5: the tolerance keeps
    /// the plain rounded guess, as before the rule.
    #[test]
    fn untrimmed_covers_keep_the_rounded_guess_at_four_fifths() {
        for short in 400..=1600usize {
            for m in [f64::round, f64::floor, f64::ceil] {
                let g = m(short as f64 * 0.8) * 1600.0 / 1280.0;
                assert_eq!(snap(g, 1600, 1.25), g.round() as usize, "{short}");
            }
        }
    }

    #[test]
    fn snap_basics() {
        assert_eq!(snap(1066.25, 1600, 1.25), 1066);
        assert_eq!(snap(1065.8, 1600, 1.25), 1065);
        assert_eq!(snap(1063.7, 1600, 1.48), 1065);
        assert_eq!(snap(1063.7, 1600, 1.25), 1064);
        assert_eq!(snap(1067.5, 1600, 1.25), 1068);
        assert_eq!(snap(1601.0, 1600, 1.25), 1600);
        assert_eq!(snap(2.0, 1600, 1.25), 8);
        assert_eq!(snap_cover_short(1067.5, 1600, 1, 1.0), 1068);
        assert_eq!(snap_cover_short(1067.5, 1600, 8, 4.0), 1064);
    }
}
