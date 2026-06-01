// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase E.2: cross-arch determinism (#8).
//
// The Phasm phase-6 H.264 encoder is supposed to produce
// byte-identical output across architectures: same fuzz YUV →
// same Annex-B bytes on macOS-arm64 / linux-x86_64 / linux-arm64.
//
// This file pins the SHA-256 hash of the transcode output for a
// canonical synthetic YUV. The hash recorded in the test was
// computed on the maintainer's macOS arm64 Apple Silicon machine.
// When the test runs on a different architecture in CI:
//   - If the hash matches → cross-arch determinism holds.
//   - If the hash differs → there's a real determinism bug in the
//     encoder kernels (almost certainly an f64 / FMA / floating-
//     point pitfall that escaped the det_math discipline).
//
// On a hash divergence: do NOT silently update the pinned hash
// without root-causing the divergence. The whole point of the
// pin is to catch determinism regressions.

#![cfg(feature = "h264-encoder")]

use phasm_core::{transcode_yuv_to_baseline_cavlc_h264, BaselineTranscodeConfig};
use sha2::{Digest, Sha256};

/// Generate a deterministic synthetic YUV sequence. Uses a small
/// LCG to fill the planes with a content-rich pattern (not flat,
/// not all-zero — would mask sign-bit + coefficient determinism
/// issues).
///
/// Same fn as in baseline_transcode.rs `tests::deterministic_yuv`,
/// duplicated here so this test is self-contained.
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

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    digest.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Phase E.2 / Task #8 sign-off gate: pinned-hash determinism.
///
/// Produces a known-input transcode output. SHA-256 of the output
/// MUST match the hex string below on every architecture (after
/// the pin is recorded on the maintainer's machine). Any divergence
/// indicates a real determinism bug in the encoder kernels.
///
/// **How to update on legitimate encoder change:** when an encoder
/// commit intentionally changes output (rare; examples: spec-bug
/// fix, deliberate algorithm tweak), update the `EXPECTED_*` hash.
/// The `MAINTAINER_RECORDED_HASH` comment block below is the
/// audit trail for hash provenance.
///
/// MAINTAINER_RECORDED_HASH:
///   - Original: 2026-04-27 / `64d112…b12825` (macOS arm64).
///   - **Re-pinned 2026-04-30** following the encoder change ramp
///     §6E-A1..§6F.2(k) (B-frame DPB, multi-ref P SPS bump, MVD
///     sign-override CABAC primitive, stealth-weighted allocation).
///     Several of these intentionally changed encoder output
///     (most notably §6E-B(a) `1d61924` — `max_num_ref_frames 1→2`
///     which alters the SPS NAL bytes for ALL transcode output).
///   - Date: 2026-04-30
///   - Architecture: macOS arm64 (Apple Silicon)
///   - Rust toolchain: stable
///   - Build profile: dev (cargo test)
///   - Output produced by: transcode_yuv_to_baseline_cavlc_h264
///     on `deterministic_yuv(32, 32, 5)` with default config.
#[test]
fn h264_transcode_deterministic_hash_32x32x5() {
    // Same canonical input across arches.
    let yuv = deterministic_yuv(32, 32, 5);
    let cfg = BaselineTranscodeConfig::defaults(32, 32, 5);
    let bytes = transcode_yuv_to_baseline_cavlc_h264(&yuv, cfg)
        .expect("transcode must succeed on canonical synthetic input");
    let hash = sha256_hex(&bytes);
    eprintln!("CROSS-ARCH-DETERMINISM HASH (32x32x5): {hash}  ({} bytes)", bytes.len());

    // Pin re-recorded 2026-04-30 on macOS arm64. Cross-arch
    // verification (x86_64 / linux-arm64 / WASM) happens in CI;
    // any divergence on a different arch indicates a real
    // determinism bug in encoder kernels.
    //
    // V0.4.A (2026-05-23) — pin re-recorded after intervening pure-Rust
    // encoder changes (#549 partition_id fix, #540 wire_only port,
    // #319/#324/#325 v1.6/v1.7 work, etc.). The V0.4.A.1 OH264 shim
    // change is unrelated to this hash — `transcode_yuv_to_baseline_
    // cavlc_h264` exercises the pure-Rust encoder, not OH264. The A.1
    // change has since been reverted as a stealth-negative (see
    // memory/h264_v04a_multi_ref_p_negative.md), but the hash stays.
    const EXPECTED_HASH: &str =
        "3f78a8af7a0428752248d553770cf455401a7a4d9f2964ba5a4b1be532619b43";
    assert_eq!(hash, EXPECTED_HASH,
        "Phase E.2 cross-arch determinism: hash mismatch (output is {} bytes)",
        bytes.len());
}

/// Same gate at a smaller fixture (single I-frame). Provides a
/// faster-converging signal for any determinism regression.
#[test]
fn h264_transcode_deterministic_hash_16x16x1() {
    let yuv = deterministic_yuv(16, 16, 1);
    let cfg = BaselineTranscodeConfig::defaults(16, 16, 1);
    let bytes = transcode_yuv_to_baseline_cavlc_h264(&yuv, cfg)
        .expect("transcode 16x16 single IDR");
    let hash = sha256_hex(&bytes);
    eprintln!("CROSS-ARCH-DETERMINISM HASH (16x16x1): {hash}  ({} bytes)", bytes.len());

    // Re-pinned 2026-04-30 (same audit trail as the 32x32x5 test).
    // V0.4.A (2026-05-23) — pin re-recorded after intervening pure-Rust
    // encoder changes (unrelated to V0.4.A.1 OH264 shim, since reverted).
    const EXPECTED_HASH: &str =
        "a1fe3c65e8265d45a08011e045a46b8ac726d2450d7379f2c1cd22bf394b2d91";
    assert_eq!(hash, EXPECTED_HASH,
        "Phase E.2 cross-arch determinism: 16x16x1 hash mismatch");
}

/// Sanity: two independent transcodes of the same input on the
/// SAME machine produce identical output. Catches in-process
/// non-determinism (uninitialized memory, hashmap iteration order,
/// thread-scheduling-dependent reductions, etc.) without requiring
/// CI cross-arch coverage.
#[test]
fn h264_transcode_intra_run_deterministic() {
    let yuv = deterministic_yuv(32, 32, 3);
    let cfg = BaselineTranscodeConfig::defaults(32, 32, 3);
    let a = transcode_yuv_to_baseline_cavlc_h264(&yuv, cfg).unwrap();
    let cfg = BaselineTranscodeConfig::defaults(32, 32, 3);
    let b = transcode_yuv_to_baseline_cavlc_h264(&yuv, cfg).unwrap();
    assert_eq!(a, b, "two transcodes of the same input must be byte-identical");
}

/// Phase 6F.2 (#68) — 1080p single-run determinism on real-world
/// iPhone YUV. Validates that the encoder is deterministic at
/// production scale, not just on tiny synthetic fixtures.
///
/// `#[ignore]` because:
/// - Reads `/tmp/img4138_1080p_f10.yuv` (regen-on-demand from the
///   gitignored `IMG_4138.MOV` source via
///   `core/test-vectors/video/h264/real-world/source/regen.sh`).
/// - Encoder takes ~3s per 1080p frame single-threaded; one-frame
///   intra-run check is the smoke gate, not for casual `cargo test`.
///
/// Run with `cargo test --features h264-encoder --test
/// h264_cross_arch_determinism -- --ignored`.
#[test]
#[ignore = "1080p smoke; needs /tmp/img4138_1080p_f10.yuv from regen.sh; ~3s/frame"]
fn h264_transcode_intra_run_deterministic_1080p() {
    let yuv_path = "/tmp/img4138_1080p_f10.yuv";
    let yuv = std::fs::read(yuv_path)
        .expect("regen /tmp/img4138_1080p_f10.yuv via test-vectors/.../regen.sh");
    let frame_size = (1920 * 1072 * 3 / 2) as usize;
    let one_frame = &yuv[..frame_size];
    let cfg = BaselineTranscodeConfig::defaults(1920, 1072, 1);
    let a = transcode_yuv_to_baseline_cavlc_h264(one_frame, cfg).unwrap();
    let cfg = BaselineTranscodeConfig::defaults(1920, 1072, 1);
    let b = transcode_yuv_to_baseline_cavlc_h264(one_frame, cfg).unwrap();
    assert_eq!(a, b, "1080p transcode must be byte-identical between runs");
    eprintln!("1080p intra-run determinism: {} bytes (matched)", a.len());
}

/// Streaming + one-shot output match (already covered in
/// baseline_transcode tests; replicated here so the cross-arch
/// CI matrix exercises both encoder paths).
#[test]
fn h264_streaming_matches_one_shot_byte_identical() {
    use phasm_core::StreamingEncoder;

    let yuv = deterministic_yuv(32, 32, 4);
    let cfg = BaselineTranscodeConfig::defaults(32, 32, 4);
    let one_shot = transcode_yuv_to_baseline_cavlc_h264(&yuv, cfg).unwrap();

    let mut stream = StreamingEncoder::new(32, 32, Some(26), 30).unwrap();
    let frame_size = (32 * 32 * 3 / 2) as usize;
    let mut concat = Vec::new();
    for f in 0..4 {
        concat.extend_from_slice(
            &stream.push_frame(&yuv[f * frame_size..(f + 1) * frame_size]).unwrap()
        );
    }
    assert_eq!(concat, one_shot, "streaming != one-shot byte-for-byte");
}
