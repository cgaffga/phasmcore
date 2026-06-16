// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! WV.6 round-trip GATE — OH264 video primary + shadow round-trip.
//!
//! Encode a synthetic clip with a primary AND one shadow through the
//! per-GOP streaming shadow path (`h264_encode_with_shadows_streaming`
//! driven by a `SliceYuvSource`), then decode
//! BOTH via `StreamingDecodeSession` (primary passphrase recovers the
//! primary; shadow passphrase recovers the shadow). This is the gate the
//! WV.6.a refactor (per-GOP primary in shadow mode,
//! `docs/design/video/h264/oh264-wv6-streaming-shadow-unification.md`) must
//! keep green: today the primary is a whole-video `frame_bits` STC; after
//! WV.6.a it is a per-GOP `chunk_frame`. Either way both messages must come
//! back. (No prior OH264 *video* shadow round-trip test existed — the older
//! shadow_roundtrip.rs covers IMAGE shadows only.)
//!
//! `#[ignore]`d by default — the whole-video shadow encode runs the OH264
//! encoder over synthetic YUV (~3-6 s) and wants `--test-threads=1`. Run:
//!   cargo test -p phasm-core --features h264-encoder --test \
//!     oh264_video_shadow_roundtrip -- --ignored --test-threads=1

#![cfg(feature = "h264-encoder")]

use phasm_core::codec::h264::openh264_stego::{
    h264_encode_with_shadows_streaming, EncodeOpts, SliceYuvSource,
};
use phasm_core::codec::h264::streaming_session::StreamingDecodeSession;
use phasm_core::codec::h264::stego::CostWeights;
use phasm_core::stego::shadow_layer::ShadowLayer;

/// Same deterministic textured YUV generator as the #530 repro.
fn synth_yuv(width: u32, height: u32, frame_idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = Vec::with_capacity((width * height) as usize);
    for j in 0..height {
        for i in 0..width {
            y.push(((i + frame_idx * 2) ^ (j + frame_idx * 3)) as u8);
        }
    }
    let (half_w, half_h) = (width / 2, height / 2);
    let mut u = Vec::with_capacity((half_w * half_h) as usize);
    let mut v = Vec::with_capacity((half_w * half_h) as usize);
    let mut s: u32 = 0xCAFE_F00D ^ frame_idx;
    for j in 0..half_h {
        for i in 0..half_w {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let pos = (i + j + frame_idx) as u8;
            u.push(((s >> 16) as u8).wrapping_add(pos));
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(((s >> 16) as u8).wrapping_add(pos));
        }
    }
    (y, u, v)
}

fn run_shadow_roundtrip(
    width: u32,
    height: u32,
    gop_size: u32,
    n_frames: u32,
    primary_msg: &str,
    primary_pass: &str,
    shadow_msg: &str,
    shadow_pass: &str,
) {
    // Assemble the whole-clip tight-I420 buffer (same synthetic content as
    // before) and drive the per-GOP streaming shadow encoder through a
    // `SliceYuvSource` — the byte-identical stand-in for the bridge's real
    // pull source. WV.6.g retired the push session; the streaming pull path
    // is the only shadow-video encode entry now.
    let frame_bytes = (width * height * 3 / 2) as usize;
    let mut yuv = Vec::with_capacity(frame_bytes * n_frames as usize);
    for f in 0..n_frames {
        let (y, u, v) = synth_yuv(width, height, f);
        yuv.extend_from_slice(&y);
        yuv.extend_from_slice(&u);
        yuv.extend_from_slice(&v);
    }
    // Primary must be the LARGEST payload so the n-shadow auto-sort doesn't
    // swap it with a shadow (mirrors shadow_roundtrip.rs's image gate).
    let shadows = [ShadowLayer {
        message: shadow_msg,
        passphrase: shadow_pass,
        files: &[],
    }];
    let opts = EncodeOpts { qp: 26, intra_period: gop_size as i32 };
    let mut src = SliceYuvSource::new(&yuv, width, height, n_frames, gop_size);
    let annex_b = h264_encode_with_shadows_streaming(
        &mut src,
        width,
        height,
        n_frames,
        opts,
        primary_msg,
        &[],
        primary_pass,
        &shadows,
        &CostWeights::default(),
        None,
    )
    .expect("streaming shadow encode");
    assert!(!annex_b.is_empty(), "empty Annex-B");

    // Primary passphrase → primary message.
    let mut dp = StreamingDecodeSession::create(primary_pass).expect("primary decode session");
    dp.push_annex_b(&annex_b).expect("push annex-b (primary)");
    let rp = dp.finish().expect("primary decode");
    assert_eq!(rp.text, primary_msg, "primary message mismatch");

    // Shadow passphrase → shadow message.
    let mut ds = StreamingDecodeSession::create(shadow_pass).expect("shadow decode session");
    ds.push_annex_b(&annex_b).expect("push annex-b (shadow)");
    let rs = ds.finish().expect("shadow decode");
    assert_eq!(rs.text, shadow_msg, "shadow message mismatch");

    eprintln!(
        "OH264 video shadow round-trip GREEN: {width}x{height} x {n_frames}f gop={gop_size} \
         (primary {} B, shadow {} B, {} annex_b bytes)",
        primary_msg.len(),
        shadow_msg.len(),
        annex_b.len()
    );
}

#[test]
#[ignore = "OH264 shadow encode ~5s; run with --ignored --test-threads=1"]
fn oh264_video_primary_plus_one_shadow_roundtrip() {
    // 320x240 x 20f @ gop=10 → 2 GOPs. Primary longer than the shadow.
    run_shadow_roundtrip(
        320, 240, /*gop=*/ 10, /*n=*/ 20,
        "primary message, longer than the shadow", "primary-pass",
        "shdw", "s",
    );
}
