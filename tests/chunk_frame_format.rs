// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
//
// #800 — chunk_frame wire-format contract (payload_len field).
//
// These exercise the build/parse edge cases that the real-fixture gate
// (mode_b_cascade_diag) does NOT reach: the extended u32 length form,
// empty chunks, and the length-strict truncation rejection that is the
// core of the m_total-ambiguity fix.
//
// Lives as an INTEGRATION test (not a #[cfg(test)] unit test in
// chunk_frame.rs) because the lib unit-test target currently fails to
// build under edition 2024 — several unrelated modules (encoder.rs B-RDO
// harness, fft2d_simd) have pre-#800 edition-migration gaps. An
// integration test compiles the lib as a dependency, sidestepping that.
// The mirror unit tests in chunk_frame.rs become runnable once the
// lib-test build debt is cleared (tracked separately).

#![cfg(feature = "video")]

use phasm_core::codec::h264::stego::chunk_frame::{
    build_chunk_frame, parse_chunk_frame, CHUNK_HEADER_LEN, CHUNK_HEADER_LEN_MAX, LEN_SENTINEL,
};

#[test]
fn roundtrip_small_inline() {
    let payload = b"the exact length matters";
    let framed = build_chunk_frame(0, 1, payload).unwrap();
    assert_eq!(framed.len(), CHUNK_HEADER_LEN + payload.len());
    let (idx, total, slice) = parse_chunk_frame(&framed).unwrap();
    assert_eq!((idx, total), (0, 1));
    assert_eq!(slice, payload);
}

#[test]
fn roundtrip_empty_chunk() {
    // payload_len=0 must be a valid inline value (the reason the sentinel
    // is 0xFFFF, not 0x0000) — empty tail chunks are routine.
    let framed = build_chunk_frame(3, 7, b"").unwrap();
    assert_eq!(framed.len(), CHUNK_HEADER_LEN);
    let (idx, total, slice) = parse_chunk_frame(&framed).unwrap();
    assert_eq!((idx, total), (3, 7));
    assert!(slice.is_empty());
}

#[test]
fn inline_to_extended_boundary() {
    let inline = vec![0xABu8; (LEN_SENTINEL as usize) - 1]; // 65534 → inline
    let f_in = build_chunk_frame(0, 1, &inline).unwrap();
    assert_eq!(f_in.len(), CHUNK_HEADER_LEN + inline.len());
    assert_eq!(parse_chunk_frame(&f_in).unwrap().2, &inline[..]);

    let escaped = vec![0xCDu8; LEN_SENTINEL as usize]; // 65535 → extended
    let f_ex = build_chunk_frame(0, 1, &escaped).unwrap();
    assert_eq!(f_ex.len(), CHUNK_HEADER_LEN_MAX + escaped.len());
    assert_eq!(parse_chunk_frame(&f_ex).unwrap().2, &escaped[..]);
}

#[test]
fn roundtrip_extended_u32() {
    let payload = vec![0x5Au8; 70_000]; // > u16 → extended length form
    let framed = build_chunk_frame(0, 1, &payload).unwrap();
    assert_eq!(framed.len(), CHUNK_HEADER_LEN_MAX + payload.len());
    let (idx, total, slice) = parse_chunk_frame(&framed).unwrap();
    assert_eq!((idx, total), (0, 1));
    assert_eq!(slice.len(), 70_000);
    assert_eq!(slice, &payload[..]);
}

#[test]
fn parse_is_length_strict() {
    // THE #800 invariant: a frame missing even one payload byte must NOT
    // parse — this is what makes the decoder reject too-small m_total
    // candidates and land on the encoder's exact m_total.
    let framed = build_chunk_frame(0, 1, b"exact length required").unwrap();
    assert!(parse_chunk_frame(&framed[..framed.len() - 1]).is_none());
    // The full frame still parses.
    assert!(parse_chunk_frame(&framed).is_some());
}

#[test]
fn parse_ignores_trailing_returns_exact_payload() {
    // A longer buffer returns exactly payload_len (a larger m_total in
    // the same w-class still recovers the right payload).
    let payload = b"abc";
    let mut framed = build_chunk_frame(0, 1, payload).unwrap();
    framed.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    assert_eq!(parse_chunk_frame(&framed).unwrap().2, payload);
}

#[test]
fn parse_rejects_false_sentinel() {
    // 0xFFFF sentinel followed by a u32 < LEN_SENTINEL is malformed.
    let bad = vec![0, 0, 0, 1, 0xFF, 0xFF, 0, 0, 0, 10, 1, 2, 3];
    assert!(parse_chunk_frame(&bad).is_none());
}
