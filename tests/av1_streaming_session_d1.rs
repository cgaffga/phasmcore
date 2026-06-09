// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! D.1 — Av1Streaming{Encode,Decode}Session round-trip.
//!
//! Exercises the Phase D session API at its D.1 scope:
//! single-frame-per-GOP internals (gop_size=1) with chunk_frame
//! header on every GOP. Round-trips a multi-frame message across N
//! sessions and asserts byte-for-byte payload recovery.
//!
//! D.2+ adds multi-frame GOPs once the phasm-rav1e fork exposes
//! `new_inter_frame` publicly. The session API stays the same;
//! only the per-GOP internals change.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;

use phasm_core::codec::av1::stego::session::{
    Av1StreamingDecodeSession, Av1StreamingEncodeParams, Av1StreamingEncodeSession,
};

const W: u32 = 256;
const H: u32 = 144;
const Q: usize = 30;
const SEEK_S: f32 = 1.0;
const SOURCE: &str = "IMG_4138.MOV";

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv420_frame(seek_s: f32) -> Vec<u8> {
    let src = corpus_root().join(SOURCE);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!("scale={W}:{H}:force_original_aspect_ratio=disable");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args([
            "-frames:v",
            "1",
            "-vf",
            &vf,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .expect("ffmpeg launch");
    assert!(out.status.success(), "ffmpeg yuv extract failed");
    let expected = (W * H * 3 / 2) as usize;
    assert_eq!(out.stdout.len(), expected);
    out.stdout
}

#[test]
fn d1_single_gop_session_round_trip() {
    // Single-frame session — D.1's most basic test. total_frames_hint=1
    // produces total_chunks=1; the payload fits in one GOP without
    // splitting.
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260604");
    }

    let message = b"phasm AV1 D.1 single-GOP session round-trip";
    let passphrase = "d1-session-2026-06-04";

    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: 1,
    };
    let yuv = extract_yuv420_frame(SEEK_S);

    let mut session = Av1StreamingEncodeSession::create(passphrase, message, params).unwrap();
    assert_eq!(session.total_chunks(), 1);

    let mut out = Vec::new();
    session.push_frame(&yuv, &mut out).unwrap();
    session.finish(&mut out).unwrap();
    assert!(!out.is_empty(), "session.finish produced empty output");

    eprintln!(
        "[D.1] single-GOP: msg_bytes={} stego_bytes={}",
        message.len(),
        out.len()
    );

    let mut decode = Av1StreamingDecodeSession::create(passphrase);
    decode.push_bytes(&out);
    let plaintext = decode.finish().unwrap();
    assert_eq!(plaintext, message);
}

#[test]
fn d1_multi_gop_session_round_trip_3_frames() {
    // 3-frame session — each frame is its own GOP (gop_size=1).
    // total_chunks=3, each GOP carries a chunk_idx 0/1/2 of the
    // pre-split payload. Decode reassembles and decrypts.
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260604");
    }

    let message = b"phasm AV1 D.1 three-GOP session - chunk reassembly across the wire";
    let passphrase = "d1-session-3gop-2026-06-04";

    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: 3,
    };

    let mut session = Av1StreamingEncodeSession::create(passphrase, message, params).unwrap();
    assert_eq!(session.total_chunks(), 3);

    let mut full_stego = Vec::new();
    let mut per_gop_sizes = Vec::new();
    for (i, seek_s) in [0.5_f32, 1.0, 1.5].iter().enumerate() {
        let yuv = extract_yuv420_frame(*seek_s);
        let prev = full_stego.len();
        session.push_frame(&yuv, &mut full_stego).unwrap();
        let gop_bytes = full_stego.len() - prev;
        per_gop_sizes.push(gop_bytes);
        eprintln!("[D.1] GOP {i} (seek {seek_s:.1}s): {gop_bytes} bytes");
    }
    session.finish(&mut full_stego).unwrap();
    assert!(per_gop_sizes.iter().all(|&n| n > 0));

    // D.4 OBU walker: caller can push the whole stream in ONE call,
    // the walker discovers per-GOP boundaries via sequence_header_obu.
    let mut decode = Av1StreamingDecodeSession::create(passphrase);
    decode.push_bytes(&full_stego);
    let plaintext = decode.finish().unwrap();
    assert_eq!(plaintext, message);
}

#[test]
fn d6_create_rejects_gop_size_zero() {
    // D.6 (2026-06-05): gop_size > 1 is supported; only 0 rejects.
    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 0,
        total_frames_hint: 2,
    };
    match Av1StreamingEncodeSession::create("p", b"m", params) {
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("gop_size"),
                "expected gop_size error, got {msg}"
            );
        }
        Ok(_) => panic!("expected error for gop_size=0"),
    }
}

#[test]
fn d1_create_rejects_total_frames_hint_zero() {
    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: 0,
    };
    match Av1StreamingEncodeSession::create("p", b"m", params) {
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("total_frames_hint"),
                "expected total_frames_hint error, got {msg}"
            );
        }
        Ok(_) => panic!("expected error for total_frames_hint=0"),
    }
}

/// D.4 OBU walker — decode session accepts arbitrary byte granularity
/// and the walker discovers per-GOP boundaries via `sequence_header_obu`.
#[test]
fn d4_obu_walker_accepts_arbitrary_byte_granularity() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260604");
    }
    let message = b"D.4 OBU walker test - arbitrary push granularity";
    let passphrase = "d4-walker-2026-06-04";

    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: 3,
    };
    let mut session = Av1StreamingEncodeSession::create(passphrase, message, params).unwrap();
    let mut full_stego = Vec::new();
    for seek_s in [0.5_f32, 1.0, 1.5] {
        let yuv = extract_yuv420_frame(seek_s);
        session.push_frame(&yuv, &mut full_stego).unwrap();
    }
    session.finish(&mut full_stego).unwrap();
    let total_bytes = full_stego.len();

    // (a) Single big push — the natural shape.
    let mut decode_single = Av1StreamingDecodeSession::create(passphrase);
    decode_single.push_bytes(&full_stego);
    assert_eq!(decode_single.finish().unwrap(), message);

    // (b) Many tiny pushes — split into 256-byte chunks at arbitrary
    // positions (not aligned to any AV1 structure). Walker must
    // reassemble + discover boundaries.
    let mut decode_chunked = Av1StreamingDecodeSession::create(passphrase);
    for chunk in full_stego.chunks(256) {
        decode_chunked.push_bytes(chunk);
    }
    assert_eq!(decode_chunked.finish().unwrap(), message);

    // (c) Byte-at-a-time push — pathological but should also work.
    let mut decode_byte = Av1StreamingDecodeSession::create(passphrase);
    for b in &full_stego {
        decode_byte.push_bytes(&[*b]);
    }
    assert_eq!(decode_byte.finish().unwrap(), message);

    eprintln!(
        "[D.4] walker round-trips {total_bytes} bytes via (a) 1 push, (b) chunked 256-byte pushes, \
         (c) byte-at-a-time pushes"
    );
}

/// D.3 V6 invariant — each GOP slab starts with `sequence_header_obu`
/// (OBU type 1). This is the boundary marker D.4's
/// `split_av1_into_gops` walker will use to split accumulated bytes
/// into per-GOP slabs at decode time. Verifies the encode side emits
/// the marker for every push_frame, which is the precondition for
/// the walker to work.
///
/// OBU header byte format (av1-spec § 5.3.2):
///   bit 7: obu_forbidden_bit = 0
///   bits 3-6: obu_type (1 = OBU_SEQUENCE_HEADER, 6 = OBU_FRAME)
///   bit 2: obu_extension_flag
///   bit 1: obu_has_size_field
///   bit 0: obu_reserved = 0
///
/// For our rav1e encode, obu_extension=0, obu_has_size_field=1,
/// so the byte is `(obu_type << 3) | 0b0000_0010`. Sequence header
/// byte = `(1 << 3) | 2 = 0b0000_1010 = 0x0A`.
const OBU_SEQUENCE_HEADER_BYTE: u8 = 0x0A;

#[test]
fn d3_each_gop_starts_with_sequence_header_obu() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260604");
    }
    let message = b"D.3 V6 invariant - each GOP slab starts with sequence_header_obu";
    let passphrase = "d3-v6-invariant-2026-06-04";

    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: 3,
    };
    let mut session = Av1StreamingEncodeSession::create(passphrase, message, params).unwrap();

    let mut per_gop_slabs: Vec<Vec<u8>> = Vec::new();
    for seek_s in [0.5_f32, 1.0, 1.5] {
        let yuv = extract_yuv420_frame(seek_s);
        let mut out = Vec::new();
        session.push_frame(&yuv, &mut out).unwrap();
        per_gop_slabs.push(out);
    }
    session.finish(&mut Vec::new()).unwrap();

    for (i, slab) in per_gop_slabs.iter().enumerate() {
        assert!(
            !slab.is_empty(),
            "GOP {i} slab is empty — session.push_frame produced nothing",
        );
        assert_eq!(
            slab[0], OBU_SEQUENCE_HEADER_BYTE,
            "GOP {i} slab does NOT start with sequence_header_obu (byte 0 = 0x{:02x}, expected 0x{:02x}). \
             This breaks D.4's OBU walker — every GOP MUST start with a sequence header so the decoder \
             can recover boundaries from concatenated bytes.",
            slab[0], OBU_SEQUENCE_HEADER_BYTE
        );
        eprintln!(
            "[D.3] GOP {i}: {} bytes, starts with 0x{:02x} (OBU_SEQUENCE_HEADER) ✓",
            slab.len(),
            slab[0]
        );
    }
}

#[test]
fn d1_finish_rejects_underflow() {
    // Hint says 3 frames but caller only pushes 1.
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260604");
    }

    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: 3,
    };
    let mut session =
        Av1StreamingEncodeSession::create("p", b"short msg", params).unwrap();
    let yuv = extract_yuv420_frame(SEEK_S);
    let mut out = Vec::new();
    session.push_frame(&yuv, &mut out).unwrap();
    match session.finish(&mut out) {
        Err(e) => {
            let msg = format!("{e:?}");
            // Phase G.0 renamed the error string from "...expected N"
            // to "...total_chunks=N / stego window..." but semantics
            // are unchanged (finish before all GOPs drained).
            assert!(
                msg.contains("expected 3")
                    || msg.contains("expected")
                    || msg.contains("stego window"),
                "expected finish underflow error, got {msg}"
            );
        }
        Ok(_) => panic!("expected underflow error"),
    }
}
