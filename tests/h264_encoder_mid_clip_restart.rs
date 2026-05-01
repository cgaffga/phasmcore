// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §long-form-stego Phase 1 — encoder mid-clip restart cover-position
// parity.
//
// The per-GOP-replay CoverFetch adapter (Phase 4) needs the encoder
// to be runnable on arbitrary GOP sub-ranges of the input YUV with
// **bit-exact equivalent COVER POSITIONS** (the data the
// PositionLoggerHook emits) for the GOPs being replayed. Bit-exact
// byte-stream identity is NOT required — at restart-IDR the
// encoder re-emits SPS+PPS+AUD parameter sets (spec-allowed) so
// raw byte-streams diverge by O(parameter-set-size) bytes per GOP.
// The cover positions, however, are slice-internal and must match.
//
// Invariant: at an IDR boundary the encoder spec-resets `frame_num`,
// `pic_order_cnt`, and the DPB. The only external priming the
// per-GOP-replay adapter must do is `stego_frame_idx` — the
// PositionKey field that keys per-frame stego positions to the
// global encode-order index.

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::stego::encoder_hook::PositionLoggerHook;
use phasm_core::codec::h264::stego::orchestrate::GopCover;

fn deterministic_yuv(w: u32, h: u32, n_frames: usize) -> Vec<u8> {
    let frame_size = (w * h * 3 / 2) as usize;
    let mut out = Vec::with_capacity(frame_size * n_frames);
    let mut s: u32 = 0x1234_5678;
    for _ in 0..n_frames {
        for _ in 0..frame_size {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            out.push((s >> 16) as u8);
        }
    }
    out
}

fn build_encoder(width: u32, height: u32) -> Encoder {
    let mut enc = Encoder::new(width, height, Some(26)).expect("encoder");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = false;
    enc.enable_mvd_stego_hook = true;
    enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
    enc
}

/// Drain the encoder's PositionLoggerHook and return the
/// accumulated `GopCover`. Encoder-side helper that mirrors what
/// `encode_pixels.rs::drain_position_logger` does internally.
fn drain_cover(enc: &mut Encoder) -> GopCover {
    let mut hook = enc.take_stego_hook().expect("hook present");
    hook.take_cover_if_logger().expect("PositionLoggerHook")
}

/// Encode `[frame_start..frame_end)` of `yuv` with `gop_size`-period
/// IDRs (the IDR for the first frame in the range is mandatory and
/// auto-emitted). Returns the accumulated cover bits + positions
/// per-domain via the PositionLoggerHook drain.
fn cover_for_frame_range(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    frame_start: usize,
    frame_end: usize,
) -> GopCover {
    let frame_size = (width * height * 3 / 2) as usize;
    let mut enc = build_encoder(width, height);
    enc.stego_frame_idx = frame_start as u32;
    assert!(frame_start < frame_end);
    assert!(frame_end <= n_frames);
    for fi in frame_start..frame_end {
        let frame = &yuv[fi * frame_size..(fi + 1) * frame_size];
        let is_idr_in_full = fi % gop_size == 0;
        let is_first_in_partial = fi == frame_start;
        let is_idr = is_idr_in_full || is_first_in_partial;
        if is_idr {
            // Encoder requires an IDR at the first frame of any
            // fresh encode. For the partial-restart case (Phase 1
            // contract), `frame_start` is always a GOP boundary so
            // is_idr_in_full == is_first_in_partial.
            enc.encode_i_frame(frame).expect("i-frame");
        } else {
            enc.encode_p_frame(frame).expect("p-frame");
        }
    }
    drain_cover(&mut enc)
}

/// Slice helper for `DomainBits` — returns the (bits, positions)
/// pair for cover indices `[lo..hi)`.
fn slice_domain(
    cov: &phasm_core::codec::h264::stego::inject::DomainBits,
    lo: usize,
    hi: usize,
) -> (Vec<u8>, Vec<phasm_core::codec::h264::stego::hook::PositionKey>) {
    (cov.bits[lo..hi].to_vec(), cov.positions[lo..hi].to_vec())
}

/// Find the cover-bit offset where frames in
/// `[frame_start..frame_end)` begin within a `full` GopCover that
/// spans [0..n_frames). Done by counting positions from frames
/// before `frame_start` — but `GopCover.positions` is keyed by
/// `frame_idx`, so we filter by that.
///
/// Returns the per-domain (lo, hi) cover-bit indices for the
/// frame range.
fn frame_range_cover_indices(
    full: &GopCover,
    frame_start: u32,
    frame_end: u32,
) -> [(usize, usize); 4] {
    use phasm_core::codec::h264::stego::inject::DomainBits;
    let domains: [&DomainBits; 4] = [
        &full.cover.coeff_sign_bypass,
        &full.cover.coeff_suffix_lsb,
        &full.cover.mvd_sign_bypass,
        &full.cover.mvd_suffix_lsb,
    ];
    let mut out = [(0usize, 0usize); 4];
    for (d, dom) in domains.iter().enumerate() {
        let mut lo = dom.positions.len();
        let mut hi = 0;
        for (i, p) in dom.positions.iter().enumerate() {
            let f = p.frame_idx();
            if f >= frame_start && f < frame_end {
                if i < lo {
                    lo = i;
                }
                if i + 1 > hi {
                    hi = i + 1;
                }
            }
        }
        // Empty range: lo > hi after the loop. Normalize.
        if lo > hi {
            lo = 0;
            hi = 0;
        }
        out[d] = (lo, hi);
    }
    out
}

#[test]
fn cover_positions_match_at_each_gop_restart() {
    // 9 frames at 64x48 with gop_size=3 → 3 GOPs, IDRs at frames
    // 0, 3, 6.
    let width = 64u32;
    let height = 48u32;
    let n_frames = 9usize;
    let gop_size = 3usize;
    let yuv = deterministic_yuv(width, height, n_frames);

    // Full-clip cover.
    let full = cover_for_frame_range(
        &yuv, width, height, n_frames, gop_size, 0, n_frames,
    );

    // Restart at each non-zero GOP boundary and confirm the
    // partial-range cover positions match the corresponding slice
    // of the full-clip cover.
    for gop_start in 1..(n_frames / gop_size) {
        let frame_start = gop_start * gop_size;
        let partial = cover_for_frame_range(
            &yuv, width, height, n_frames, gop_size, frame_start, n_frames,
        );

        let indices = frame_range_cover_indices(
            &full,
            frame_start as u32,
            n_frames as u32,
        );
        let domains = [
            ("coeff_sign", &full.cover.coeff_sign_bypass, &partial.cover.coeff_sign_bypass),
            ("coeff_suffix", &full.cover.coeff_suffix_lsb, &partial.cover.coeff_suffix_lsb),
            ("mvd_sign", &full.cover.mvd_sign_bypass, &partial.cover.mvd_sign_bypass),
            ("mvd_suffix", &full.cover.mvd_suffix_lsb, &partial.cover.mvd_suffix_lsb),
        ];

        for (d, (name, full_dom, partial_dom)) in domains.iter().enumerate() {
            let (lo, hi) = indices[d];
            let (full_bits, full_positions) = slice_domain(full_dom, lo, hi);
            assert_eq!(
                full_bits.len(),
                partial_dom.bits.len(),
                "domain={name} GOP={gop_start} restart: bit-count mismatch \
                 (full slice {} vs partial {})",
                full_bits.len(),
                partial_dom.bits.len(),
            );
            assert_eq!(
                full_bits, partial_dom.bits,
                "domain={name} GOP={gop_start} restart: bit values diverge",
            );
            assert_eq!(
                full_positions, partial_dom.positions,
                "domain={name} GOP={gop_start} restart: PositionKeys diverge \
                 (PositionKey carries frame_idx + mb_addr + domain + syntax_path; \
                 mismatch implies stego_frame_idx priming or per-frame state \
                 hygiene is broken)",
            );
        }
    }
}

/// Sanity gate: cover positions from a full-clip encode are
/// invariant under repeated calls (same machine, same input).
/// Catches non-determinism that the restart test wouldn't isolate.
#[test]
fn full_clip_cover_is_deterministic_across_calls() {
    let width = 64u32;
    let height = 48u32;
    let n_frames = 6usize;
    let gop_size = 3usize;
    let yuv = deterministic_yuv(width, height, n_frames);

    let cover_a = cover_for_frame_range(
        &yuv, width, height, n_frames, gop_size, 0, n_frames,
    );
    let cover_b = cover_for_frame_range(
        &yuv, width, height, n_frames, gop_size, 0, n_frames,
    );
    assert_eq!(cover_a.cover.coeff_sign_bypass.bits, cover_b.cover.coeff_sign_bypass.bits);
    assert_eq!(cover_a.cover.coeff_sign_bypass.positions, cover_b.cover.coeff_sign_bypass.positions);
    assert_eq!(cover_a.cover.mvd_sign_bypass.bits, cover_b.cover.mvd_sign_bypass.bits);
    assert_eq!(cover_a.cover.mvd_sign_bypass.positions, cover_b.cover.mvd_sign_bypass.positions);
}
