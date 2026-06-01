// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
//
// V0.4.E regression test (2026-05-23) — verify the OH264 shim pins
// `eSpsPpsIdStrategy = CONSTANT_ID` so every PPS NAL emits with
// pps_id=0 and every slice header references pps_id=0 across all GOPs.
//
// Background: OH264's `FillDefault` (`param_svc.h:187`) sets
// `INCREASING_ID` which rotates pps_id per IDR (0,1,2,...,N). The iOS
// AVAssetWriter mux only captures the first PPS NAL into its
// CMVideoFormatDescription, leaving later GOP slices referencing
// pps_ids that no longer exist in the muxed MP4 — producing ffmpeg's
// "non-existing PPS N referenced" warnings on every IDR after the
// first, and a malformed (though still phasm-decodable) MP4.
//
// `phasm_encoder_shim.cc` (V0.4.E) overrides to CONSTANT_ID to:
//   1. Fix the mux data-integrity bug (all PPSs identical → first-PPS-
//      only capture is correct by construction).
//   2. Match the dominant smartphone-camera Layer 3 cohort (iPhone /
//      DJI / Lumix / phone-recorded x264 all emit CONSTANT_ID).
//
// This test exercises an 8-frame fixture with intra_period=2 (= 4
// IDRs) and asserts every PPS NAL and every slice header carries
// pps_id=0.

#![cfg(all(feature = "h264-encoder", feature = "openh264-backend"))]

use phasm_core::codec::h264::openh264_stego::{openh264_stego_encode_yuv_text, EncodeOpts};

fn make_yuv(w: usize, h: usize, n: usize) -> Vec<u8> {
    let y = w * h;
    let uv = (w / 2) * (h / 2);
    let frame = y + 2 * uv;
    let mut buf = vec![0u8; frame * n];
    for f in 0..n {
        let off = f * frame;
        for yy in 0..h {
            for xx in 0..w {
                buf[off + yy * w + xx] = (((xx + yy + f * 8) & 0xff) ^ ((xx ^ yy) & 0x3f)) as u8;
            }
        }
        for yy in 0..(h / 2) {
            for xx in 0..(w / 2) {
                buf[off + y + yy * (w / 2) + xx] = 128;
                buf[off + y + uv + yy * (w / 2) + xx] = 128;
            }
        }
    }
    buf
}

fn parse_ue(bits: &[u8], mut pos: usize) -> (u32, usize) {
    let mut n = 0;
    while pos < bits.len() && bits[pos] == 0 {
        n += 1;
        pos += 1;
    }
    pos += 1;
    let mut val = 1u32;
    for _ in 0..n {
        val = (val << 1) | bits[pos] as u32;
        pos += 1;
    }
    (val - 1, pos)
}

fn bits_of(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(b.len() * 8);
    for &byte in b {
        for i in 0..8 {
            out.push((byte >> (7 - i)) & 1);
        }
    }
    out
}

fn rbsp(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len());
    let mut j = 0;
    while j < payload.len() {
        if j + 2 < payload.len() && payload[j] == 0 && payload[j + 1] == 0 && payload[j + 2] == 0x03 {
            out.push(0);
            out.push(0);
            j += 3;
        } else {
            out.push(payload[j]);
            j += 1;
        }
    }
    out
}

fn split_annex_b(stego: &[u8]) -> Vec<(usize, usize)> {
    let mut nals = Vec::new();
    let mut i = 0;
    while i + 3 <= stego.len() {
        let j;
        if &stego[i..i + 3] == b"\x00\x00\x01" {
            j = i + 3;
        } else if i + 4 <= stego.len() && &stego[i..i + 4] == b"\x00\x00\x00\x01" {
            j = i + 4;
        } else {
            i += 1;
            continue;
        }
        let mut k = j + 1;
        loop {
            if k + 3 > stego.len() {
                k = stego.len();
                break;
            }
            if &stego[k..k + 3] == b"\x00\x00\x01" {
                break;
            }
            if k + 4 <= stego.len() && &stego[k..k + 4] == b"\x00\x00\x00\x01" {
                break;
            }
            k += 1;
        }
        nals.push((j, k));
        i = k;
    }
    nals
}

#[test]
fn pps_constant_id_across_multi_gop() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
    }
    let yuv = make_yuv(480, 272, 8);
    let stego = openh264_stego_encode_yuv_text(
        &yuv,
        480,
        272,
        8,
        EncodeOpts { qp: 26, intra_period: 2 },
        "verify pps fix",
        "passphrase",
    )
    .expect("encode");

    let nals = split_annex_b(&stego);
    let mut pps_ids: Vec<(u32, u32)> = Vec::new();
    let mut slice_ids: Vec<(u8, u32)> = Vec::new();
    let mut idr_count = 0;
    for &(s, e) in &nals {
        if s >= e {
            continue;
        }
        let nal_type = stego[s] & 0x1F;
        let r = rbsp(&stego[s + 1..e]);
        let bits = bits_of(&r);
        if nal_type == 8 {
            let (pps_id, p) = parse_ue(&bits, 0);
            let (sps_id, _) = parse_ue(&bits, p);
            pps_ids.push((pps_id, sps_id));
        } else if nal_type == 1 || nal_type == 5 {
            let (_, p) = parse_ue(&bits, 0);
            let (_, p) = parse_ue(&bits, p);
            let (pid, _) = parse_ue(&bits, p);
            slice_ids.push((nal_type, pid));
            if nal_type == 5 {
                idr_count += 1;
            }
        }
    }

    assert!(idr_count >= 4, "fixture should produce ≥4 IDRs to exercise PPS rotation, got {idr_count}");
    assert!(pps_ids.len() >= 4, "expected ≥4 PPS NALs (one per IDR), got {}", pps_ids.len());

    for (i, &(pid, sid)) in pps_ids.iter().enumerate() {
        assert_eq!(
            pid, 0,
            "PPS NAL #{i} has pps_id={pid}, expected 0 (CONSTANT_ID strategy). \
             If this fires, OH264 default INCREASING_ID has come back — check \
             phasm_encoder_shim.cc for the `eSpsPpsIdStrategy = CONSTANT_ID` line."
        );
        assert_eq!(sid, 0, "PPS NAL #{i} has sps_id={sid}, expected 0");
    }

    let max_slice_ref = slice_ids.iter().map(|(_, p)| *p).max().unwrap_or(0);
    assert_eq!(
        max_slice_ref, 0,
        "max slice pps_id_ref={max_slice_ref}, expected 0 (CONSTANT_ID). \
         Some slice references a non-zero PPS ID; OH264 has reverted to INCREASING_ID."
    );

    eprintln!(
        "PPS NALs={} (all pps_id=0 ✓), slices={} (all pps_id_ref=0 ✓), IDRs={}",
        pps_ids.len(),
        slice_ids.len(),
        idr_count
    );
}
