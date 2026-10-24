// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Per-GOP primary allocation: how many message bytes each GOP's chunk
//! carries.
//!
//! The encoders used to split the primary evenly
//! (`split_message_into_chunks`), so a GOP with little cover — almost always
//! the trailing partial GOP — was asked for a full share and the encode
//! failed there, after every earlier GOP had been encoded. Here each GOP gets
//! bytes in proportion to its measured cover (the H.264 allocator,
//! [`allocate_chunks_proportional`]), so a message that fits the clip fits
//! every GOP. See
//! `docs/design/video/av1/capacity-split-and-payload-limits-2026-10-23.md` § 4.
//!
//! Wire constraint: the decoder needs a chunk frame in every GOP until the
//! message is complete (GOPs after it are natural). So a GOP too poor to
//! carry even a chunk header ends the window — no message bytes go past it.

use crate::codec::av1::stego::capacity::{STC_PAYLOAD_RATIO_DEN, STC_PAYLOAD_RATIO_NUM};
use crate::codec::av1::stego::orchestrator::Av1StegoError;
use crate::stego::chunk_frame::{
    allocate_chunks_proportional, CAP2_DEFAULT_R_TARGET, CHUNK_FRAME_FIRST_HEADER_LEN_V3_1_MAX,
    CHUNK_FRAME_NEXT_HEADER_LEN_V3_1_MAX,
};

/// One GOP's primary capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GopPrimaryCap {
    /// Bytes the GOP's STC can carry, chunk header included:
    /// `(cover − wet) × STC_PAYLOAD_RATIO / 8`.
    pub raw_bytes: usize,
    /// Message bytes after the chunk header (sized for the extended header,
    /// so a chunk of any length fits), saturating at 0.
    pub payload_bytes: usize,
}

/// Extended (worst-case) chunk header length for GOP `gop_idx`.
fn header_max(gop_idx: usize) -> usize {
    if gop_idx == 0 {
        CHUNK_FRAME_FIRST_HEADER_LEN_V3_1_MAX
    } else {
        CHUNK_FRAME_NEXT_HEADER_LEN_V3_1_MAX
    }
}

/// Per-GOP capacity from measured cover bits. `wet_bits[g]` are cover
/// positions the primary STC cannot use (shadow-occupied); all zero for a
/// no-shadow encode. Panics if the slices differ in length.
pub fn gop_primary_caps(cover_bits: &[usize], wet_bits: &[usize]) -> Vec<GopPrimaryCap> {
    assert_eq!(cover_bits.len(), wet_bits.len(), "one wet count per GOP");
    cover_bits
        .iter()
        .zip(wet_bits)
        .enumerate()
        .map(|(g, (&cover, &wet))| {
            let raw_bytes =
                cover.saturating_sub(wet) * STC_PAYLOAD_RATIO_NUM / STC_PAYLOAD_RATIO_DEN / 8;
            GopPrimaryCap {
                raw_bytes,
                payload_bytes: raw_bytes.saturating_sub(header_max(g)),
            }
        })
        .collect()
}

/// Index of the first GOP that cannot carry even a chunk header, or
/// `caps.len()`. Message bytes never go at or past it.
pub fn window_end(caps: &[GopPrimaryCap]) -> usize {
    caps.iter()
        .enumerate()
        .position(|(g, c)| c.raw_bytes < header_max(g))
        .unwrap_or(caps.len())
}

/// Plan `message_len` bytes over the GOPs: proportional to capacity inside
/// the window, zero from [`window_end`] on. `Σ == message_len`.
pub fn plan_primary(caps: &[GopPrimaryCap], message_len: usize) -> Result<Vec<usize>, Av1StegoError> {
    replan_from(caps, 0, message_len).ok_or_else(|| message_too_large(caps, message_len))
}

/// The `MessageTooLarge` for a `message_len`-byte primary over `caps`:
/// available is what a plan could use — the payload capacity of the GOPs
/// before [`window_end`], since no message byte goes at or past it.
pub fn message_too_large(caps: &[GopPrimaryCap], message_len: usize) -> Av1StegoError {
    let available: usize = caps[..window_end(caps)].iter().map(|c| c.payload_bytes).sum();
    Av1StegoError::MessageTooLarge {
        needed_bits: message_len.saturating_mul(8),
        available_bits: available.saturating_mul(8),
    }
}

/// Plan `remaining` bytes over GOPs `from_gop..caps.len()` — the carry
/// after a GOP embedded less than planned. Same window rule as
/// [`plan_primary`]; `None` if `remaining` exceeds what those GOPs can hold.
/// The result has `caps.len() - from_gop` entries.
pub fn replan_from(caps: &[GopPrimaryCap], from_gop: usize, remaining: usize) -> Option<Vec<usize>> {
    let n = caps.len().saturating_sub(from_gop);
    let end = window_end(caps).max(from_gop);
    let window: Vec<usize> = caps[from_gop.min(caps.len())..end.min(caps.len())]
        .iter()
        .map(|c| c.payload_bytes)
        .collect();
    let mut plan = allocate_chunks_proportional(&window, remaining, CAP2_DEFAULT_R_TARGET)?;
    plan.resize(n, 0);
    Some(plan)
}

/// Record what GOP `gop_idx` actually carried after an infeasible embed:
/// `Some(bytes)` lowers its capacity to the chunk that fit (never raises
/// it); `None` — not even a header-only chunk fit — makes it
/// header-incapable, ending the window there.
///
/// The cover-based estimate is not exact per GOP: STC fails on runs of
/// wet (infinite-cost) positions, not on their overall share, and a
/// keyframe-only trailing GOP can fail at a density the full GOPs carry
/// easily. Re-planning with the learned caps moves the shortfall onto GOPs
/// that still have room — including EARLIER ones, which [`replan_from`]
/// cannot reach.
pub fn learn_cap(caps: &mut [GopPrimaryCap], gop_idx: usize, achieved: Option<usize>) {
    let cap = &mut caps[gop_idx];
    match achieved {
        Some(bytes) => cap.payload_bytes = cap.payload_bytes.min(bytes),
        None => *cap = GopPrimaryCap { raw_bytes: 0, payload_bytes: 0 },
    }
}

/// Next smaller chunk to try after an infeasible embed:
/// `want − max(1, want / 8)`, saturating at 0 (a header-only chunk).
pub fn shrink_step(want: usize) -> usize {
    want.saturating_sub((want / 8).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::av1::stego::orchestrator::Av1StegoError;

    #[test]
    fn caps_apply_ratio_and_extended_header() {
        let c = gop_primary_caps(&[5_000_000, 642_285], &[0, 0]);
        assert_eq!(c[0].raw_bytes, 5_000_000 * 15 / 100 / 8); // 93_750
        assert_eq!(c[0].payload_bytes, 93_750 - 14);
        assert_eq!(c[1].payload_bytes, 642_285 * 15 / 100 / 8 - 10);
    }

    #[test]
    fn wet_bits_reduce_capacity() {
        let c = gop_primary_caps(&[1_000_000], &[200_000]);
        assert_eq!(c[0].raw_bytes, 800_000 * 15 / 100 / 8);
    }

    #[test]
    fn plan_is_proportional_and_spares_the_short_tail() {
        // iphone5 shape: 12 full GOPs + a 3-frame tail; 307_645 B framed.
        let mut cover = vec![5_000_000usize; 12];
        cover.push(642_285);
        let caps = gop_primary_caps(&cover, &[0; 13]);
        let plan = plan_primary(&caps, 307_645).unwrap();
        assert_eq!(plan.iter().sum::<usize>(), 307_645);
        assert!(plan[12] <= caps[12].payload_bytes);
        assert!(plan[12] * 5 < plan[0]); // tail takes ~1/8 of a full GOP's share
    }

    #[test]
    fn plan_over_capacity_is_message_too_large() {
        let caps = gop_primary_caps(&[100_000, 100_000], &[0, 0]);
        let total: usize = caps.iter().map(|c| c.payload_bytes).sum();
        assert!(matches!(
            plan_primary(&caps, total + 1),
            Err(Av1StegoError::MessageTooLarge { .. })
        ));
    }

    #[test]
    fn plan_ends_window_before_header_incapable_gop() {
        let caps = gop_primary_caps(&[1_000_000, 100, 1_000_000], &[0, 0, 0]);
        assert_eq!(window_end(&caps), 1);
        let plan = plan_primary(&caps, 1_000).unwrap();
        assert_eq!(plan, vec![1_000, 0, 0]);
    }

    /// The error reports what a plan could use: GOPs from the first
    /// header-incapable one on never carry message bytes, so their
    /// capacity is not "available".
    #[test]
    fn message_too_large_counts_only_the_window() {
        let caps = gop_primary_caps(&[1_000_000, 100, 1_000_000], &[0, 0, 0]);
        match message_too_large(&caps, 999_999) {
            Av1StegoError::MessageTooLarge { needed_bits, available_bits } => {
                assert_eq!(needed_bits, 999_999 * 8);
                assert_eq!(available_bits, caps[0].payload_bytes * 8);
            }
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn replan_from_carries_remainder_to_later_gops() {
        let caps = gop_primary_caps(&[1_000_000; 4], &[0; 4]);
        let rest = replan_from(&caps, 2, 30_000).unwrap();
        assert_eq!(rest.len(), 2);
        assert_eq!(rest.iter().sum::<usize>(), 30_000);
        assert!(replan_from(&caps, 3, caps[3].payload_bytes + 1).is_none());
    }

    #[test]
    fn learned_cap_moves_bytes_to_earlier_gops() {
        let mut caps = gop_primary_caps(&[1_000_000; 3], &[0; 3]);
        learn_cap(&mut caps, 2, Some(100));
        assert_eq!(caps[2].payload_bytes, 100);
        let plan = plan_primary(&caps, 30_000).unwrap();
        assert!(plan[2] <= 100);
        assert_eq!(plan.iter().sum::<usize>(), 30_000);
    }

    #[test]
    fn learned_header_incapable_gop_ends_the_window() {
        let mut caps = gop_primary_caps(&[1_000_000; 3], &[0; 3]);
        learn_cap(&mut caps, 1, None);
        assert_eq!(window_end(&caps), 1);
    }

    #[test]
    fn learned_cap_never_raises_capacity() {
        let mut caps = gop_primary_caps(&[100_000], &[0]);
        let before = caps[0].payload_bytes;
        learn_cap(&mut caps, 0, Some(before + 500));
        assert_eq!(caps[0].payload_bytes, before);
    }

    #[test]
    fn shrink_step_reaches_zero() {
        assert_eq!(shrink_step(800), 700);
        assert_eq!(shrink_step(1), 0);
        assert_eq!(shrink_step(0), 0);
    }
}
