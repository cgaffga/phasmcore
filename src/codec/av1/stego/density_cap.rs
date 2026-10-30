// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Damage-aware per-GOP caps: a GOP with dense cover carries less.
//!
//! The primary plan splits the message in proportion to each GOP's cover, so
//! every GOP runs at the same α. A GOP whose cover is dense (fast motion,
//! coded mostly with intra blocks that carry large residuals) then takes
//! many more flips per pixel than a calm one, and its flips land close
//! enough together that the damage shows as blocks. On the 20-GOP IMG_4138
//! 720p clip at the 1/8 red end, the fast-motion GOPs 9–15 (1.9–4.2 cover
//! positions per pixel, summed over the GOP) each had 1 800–2 700 16×16
//! blocks under 25 dB; the GOPs at or under 1.5 had at most 274, most under
//! 120 (`docs/design/video/av1/v2-gate-results.md`, "Damage-aware per-GOP
//! caps").
//!
//! The H.264 planner bounds the predicted flips in intra MBs of P frames
//! instead (main d7793b54). In AV1 that signal does not separate the GOPs:
//! at q100, 54–95 % of every GOP's cover is in intra blocks of inter frames,
//! calm GOPs included, because rav1e's inter blocks carry little residual.
//! The density does.
//!
//! The bound applies to the cover the planner gives a GOP, so the flips
//! per pixel of a dense GOP at the red end are those of a GOP at the bound.
//! Encoder-only: the decoder reads each GOP's chunk length from its chunk
//! frame. Integer arithmetic, so the probe's figures match on every platform.

/// The most cover positions per pixel of the frame, summed over the GOP,
/// that the planner uses at full α: `NUM / DEN`.
pub const DMG_COVER_PER_PX: (usize, usize) = (3, 2);
/// A dense GOP keeps at least `1 / DMG_WEIGHT_MIN_DEN` of its cover.
pub const DMG_WEIGHT_MIN_DEN: usize = 4;

/// The GOP's quality-safe cover: its `cover` positions, at most
/// [`DMG_COVER_PER_PX`] per pixel of a `width × height` frame, at least a
/// [`DMG_WEIGHT_MIN_DEN`]th of `cover`. Never above `cover`.
pub fn quality_safe_cover(cover: usize, width: u32, height: u32) -> usize {
    let px = width as usize * height as usize;
    let (num, den) = DMG_COVER_PER_PX;
    let bound = px.saturating_mul(num) / den;
    cover.min(bound.max(cover / DMG_WEIGHT_MIN_DEN))
}

/// The positions the primary plan may use in a GOP with `cover` positions
/// of which `wet` are taken (shadow slots): `cover − wet`, scaled by the
/// GOP's weight `quality_safe_cover / cover`.
pub fn quality_safe_usable(cover: usize, wet: usize, width: u32, height: u32) -> usize {
    let usable = cover.saturating_sub(wet);
    let safe = quality_safe_cover(cover, width, height);
    if safe >= cover {
        return usable;
    }
    (usable as u128 * safe as u128 / cover as u128) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 1280;
    const H: u32 = 720;
    const PX: usize = 1280 * 720;

    #[test]
    fn cover_at_or_under_the_bound_is_unchanged() {
        assert_eq!(quality_safe_cover(0, W, H), 0);
        assert_eq!(quality_safe_cover(PX, W, H), PX);
        assert_eq!(quality_safe_cover(PX * 3 / 2, W, H), PX * 3 / 2);
    }

    #[test]
    fn dense_cover_is_cut_to_the_bound() {
        // IMG_4138 GOP 11: 3 829 561 positions at 720p (4.16 per pixel).
        assert_eq!(quality_safe_cover(3_829_561, W, H), PX * 3 / 2);
        assert_eq!(quality_safe_cover(PX * 2, W, H), PX * 3 / 2);
    }

    #[test]
    fn very_dense_cover_keeps_a_quarter() {
        assert_eq!(quality_safe_cover(PX * 10, W, H), PX * 10 / 4);
    }

    #[test]
    fn empty_frame_keeps_a_quarter() {
        assert_eq!(quality_safe_cover(1000, 0, 0), 250);
    }

    #[test]
    fn usable_scales_the_free_positions_by_the_weight() {
        // Under the bound: the free positions, as before.
        assert_eq!(quality_safe_usable(PX, 1000, W, H), PX - 1000);
        // Twice the bound: weight 3/4.
        assert_eq!(quality_safe_usable(PX * 2, 0, W, H), PX * 3 / 2);
        assert_eq!(quality_safe_usable(PX * 2, 4000, W, H), (PX * 2 - 4000) * 3 / 4);
        assert_eq!(quality_safe_usable(100, 200, W, H), 0);
    }
}
