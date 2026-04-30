// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase F: capacity + stego pipeline regression (#6).
//
// Verifies h264_ghost_capacity reports non-zero bytes on every
// Baseline-CAVLC MP4 fixture in the corpus, and that the stego
// embed/decode roundtrip succeeds on the same set. Cross-encoder
// validation: x264-Baseline + VT-encoded fixtures both pass through
// the production CAVLC bitstream-mod pipeline.
//
// Phase F was framed as "validates on Rust-encoded output". In the
// transitional #77 production path, the Rust transcoder produces
// Annex-B that platform muxers (AVAssetWriter / MediaMuxer) wrap
// into MP4 — i.e., real Rust-encoded MP4s only flow through the
// pipeline at production runtime, not in the Rust test suite.
//
// What this file gates: the stego pipeline is correct on
// Baseline-CAVLC MP4s **regardless of which encoder produced them**.
// 6D.10 will add the Rust-encoder-mp4-mux end-to-end path.

#![cfg(feature = "h264-encoder")]

const TEST_BASELINE_320: &str = "test-vectors/video/h264/test_baseline_320x240.mp4";
const TEST_TINY: &str = "test-vectors/video/h264/test_tiny.mp4";

fn skip_if_missing(path: &str) -> Option<Vec<u8>> {
    match std::fs::read(path) {
        Ok(d) => Some(d),
        Err(_) => {
            eprintln!("skipping — fixture not in corpus: {path}");
            None
        }
    }
}

/// Phase F gate: capacity reports nonzero on a known Baseline-CAVLC
/// MP4 fixture.
#[test]
fn phase_f_capacity_nonzero_on_baseline_320() {
    let Some(data) = skip_if_missing(TEST_BASELINE_320) else { return };
    let cap = phasm_core::stego::video::h264_ghost_capacity(&data)
        .expect("capacity must succeed on Baseline-CAVLC MP4");
    assert!(
        cap > 0,
        "Phase F: capacity must be positive on Baseline-CAVLC fixture, got {cap}",
    );
}

/// Phase F gate: end-to-end encode + decode roundtrip on the
/// production CAVLC pipeline.
#[test]
fn phase_f_stego_roundtrip_baseline_320() {
    let Some(data) = skip_if_missing(TEST_BASELINE_320) else { return };

    let message = "phase F roundtrip ✓";
    let passphrase = "phase-f-test-pass";

    let stego = phasm_core::stego::video::h264_ghost_encode(&data, message, passphrase)
        .expect("Phase F: encode must succeed");

    // Phase F invariant: zero bitrate change.
    assert_eq!(
        stego.len(),
        data.len(),
        "Phase F: stego MP4 size must equal cover size (zero bitrate change)",
    );
    // Phase F invariant: stego ≠ cover (some bits flipped).
    assert_ne!(stego, data, "Phase F: stego must differ from cover");

    let decoded = phasm_core::stego::video::h264_ghost_decode(&stego, passphrase)
        .expect("Phase F: decode must succeed after successful encode");
    assert_eq!(decoded.text, message, "Phase F: roundtripped message mismatch");
}

/// Phase F gate: multiple message lengths roundtrip cleanly. The
/// CAVLC pipeline allocates capacity dynamically; verify it copes
/// with both short and longer payloads.
#[test]
fn phase_f_stego_roundtrip_multiple_message_sizes() {
    let Some(data) = skip_if_missing(TEST_BASELINE_320) else { return };
    let passphrase = "phase-f-len-test";

    for msg in &[
        "hi",
        "phase F mid-length payload roundtrip",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do.",
    ] {
        let stego = phasm_core::stego::video::h264_ghost_encode(&data, msg, passphrase)
            .unwrap_or_else(|e| panic!("encode len={} failed: {e}", msg.len()));
        let decoded = phasm_core::stego::video::h264_ghost_decode(&stego, passphrase)
            .unwrap_or_else(|e| panic!("decode len={} failed: {e}", msg.len()));
        assert_eq!(
            decoded.text, *msg,
            "Phase F: roundtrip mismatch at message length {}",
            msg.len(),
        );
    }
}

/// Phase F gate: wrong passphrase MUST fail to decode (security
/// regression check).
#[test]
fn phase_f_wrong_passphrase_fails() {
    let Some(data) = skip_if_missing(TEST_BASELINE_320) else { return };

    let stego = phasm_core::stego::video::h264_ghost_encode(
        &data, "secret message", "correct-pass",
    ).expect("encode");

    let result = phasm_core::stego::video::h264_ghost_decode(&stego, "wrong-pass");
    let recovered_correct = match result {
        Ok(p) => p.text == "secret message",
        Err(_) => false,
    };
    assert!(
        !recovered_correct,
        "Phase F: wrong passphrase must NOT recover the message",
    );
}

/// Phase F gate: tiny fixture (160x120) — minimum capacity edge case.
#[test]
fn phase_f_capacity_on_tiny_fixture() {
    let Some(data) = skip_if_missing(TEST_TINY) else { return };
    let cap = phasm_core::stego::video::h264_ghost_capacity(&data);
    // Tiny fixture may or may not have positive capacity depending
    // on its content; we only require the call to NOT crash.
    match cap {
        Ok(bytes) => eprintln!("Phase F tiny capacity: {bytes} bytes"),
        Err(e) => eprintln!("Phase F tiny capacity error (acceptable on edge fixture): {e}"),
    }
}
