// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.6.1 / C.6.2 (#415 / #416) — cross-arch determinism hash-pin.
//
// Same YUV input + same passphrase + same `PHASM_DETERMINISTIC_SEED`
// should produce byte-identical Annex-B output across host
// architectures (macOS arm64 / Linux x86_64 / Linux arm64).
//
// **Test mode**: ASSERT (since 2026-05-14). The CI matrix run at
// https://github.com/cgaffga/phasmcore/actions/runs/25868497830
// confirmed all three architectures (macOS aarch64, Linux x86_64,
// Linux aarch64) produce a byte-identical 47118-byte Annex-B stream
// with SHA-256 `35fd85a4...59c1f3e`. The test now asserts this hash
// strictly; a hash mismatch on any arch is a real regression.
//
// **The empirical answer to cross-arch determinism**: phasm OH264
// stego output IS bit-identical across SSE/AVX and NEON for this
// fixture. OpenH264's SIMD kernels (`codec/encoder/core/x86/` vs
// `arm64/`) compute motion-estimation costs with different
// instruction widths; on this 480×272 × 4-frame synthetic fixture
// at QP=26, those differences either don't fire or compose to a
// byte-identical cover set. C.6.3 (fork-side SIMD non-determinism
// patches) is therefore not needed — closed as "not required".
//
// **Critical: the test must NEVER print binary content to stderr or
// stdout.** Only text (hash hex, lengths, target triple, pass/fail).
// The CI workflow this test is exercised from explicitly forbids
// artifact upload + caching, so all evidence of the OH264 binary is
// destroyed at job teardown. Printing binary into the log would
// defeat that. The test's output format is intentionally minimal:
// hash hex + size + arch tag + pass/fail.
//
// Run locally:
//   PHASM_DETERMINISTIC_SEED=42 cargo test --release \
//     --features "h264-encoder" \
//     --test openh264_cross_arch_determinism -- --nocapture

#![cfg(feature = "h264-encoder")]

mod common;
use common::oh264_stream;

use sha2::{Digest, Sha256};

/// Self-contained YUV fixture. Synthetic content so the test runs in
/// any CI environment without external corpus assets. 480×272 × 4
/// frames at I420 planar; gradient + checkerboard so the encoder has
/// non-trivial coefficients to flip stego bits into.
fn synthetic_yuv() -> Vec<u8> {
    const W: usize = 480;
    const H: usize = 272;
    const N: usize = 4;
    let y_size = W * H;
    let uv_size = (W / 2) * (H / 2);
    let frame_size = y_size + 2 * uv_size;
    let mut buf = vec![0u8; frame_size * N];
    for f in 0..N {
        let off = f * frame_size;
        // Y plane: diagonal gradient + per-frame phase shift to
        // generate motion across frames.
        for y in 0..H {
            for x in 0..W {
                let v = (((x + y + f * 8) & 0xff) ^ ((x ^ y) & 0x3f)) as u8;
                buf[off + y * W + x] = v;
            }
        }
        // U plane: vertical bands.
        for y in 0..(H / 2) {
            for x in 0..(W / 2) {
                buf[off + y_size + y * (W / 2) + x] = (128 + ((x as i32 - W as i32 / 4) / 8)) as u8;
            }
        }
        // V plane: horizontal bands.
        for y in 0..(H / 2) {
            for x in 0..(W / 2) {
                buf[off + y_size + uv_size + y * (W / 2) + x] =
                    (128 + ((y as i32 - H as i32 / 4) / 8)) as u8;
            }
        }
    }
    buf
}

#[test]
fn cross_arch_determinism_record_hash() {
    const W: u32 = 480;
    const H: u32 = 272;
    const N: u32 = 4;

    // Determinism prerequisite: `stego/crypto.rs` checks this env var
    // and uses a fixed seed for salt + AES-GCM-SIV nonce when set.
    // Without it, every encode of the same input produces different
    // bytes (correct security behaviour) but defeats hash-pinning.
    // SAFETY: set_var is single-threaded in this test setup.
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "42");
    }

    let yuv = synthetic_yuv();
    // Production 4-domain streaming session. N==4 with intra_period 4 is a
    // single GOP, so `encode` (gop_size = n_frames) reproduces the geometry.
    let stego = oh264_stream::encode(
        &yuv, W, H, N,
        26,
        "phasm cross-arch determinism v1",
        "cross-arch-pass",
    ).expect("oh264 stego encode");

    let mut hasher = Sha256::new();
    hasher.update(&stego);
    let hash = hasher.finalize();
    let hash_hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();

    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;

    // INTENTIONALLY MINIMAL OUTPUT — text only, no binary content.
    eprintln!("target={os}-{arch} stego_len={} sha256={hash_hex}", stego.len());

    // Pinned cross-arch hash. Established 2026-05-14 from a 3-arch
    // matrix CI run (macOS aarch64 + Linux x86_64 + Linux aarch64);
    // see workflow run id 25868497830. All three produced this
    // identical hash + 47118-byte stego.
    //
    // **Re-pinned 2026-05-23** following V0.4.D fork patches:
    //   1. constraint_set4/5 emit 0 instead of 1 (au_set.cpp:277-278);
    //   2. pic_order_cnt_type emit 0 instead of 2 (au_set.cpp:503).
    // Both patches close real-world cohort gaps surfaced by V0.4.C
    // SPS audit. Stream gains 9 bytes (47119 → 47128) due to the
    // per-slice pic_order_cnt_lsb field that POC type 0 mandates.
    // Cross-arch matrix re-verification still pending (workflow is
    // manual-only via workflow_dispatch); current pin captures the
    // macOS arm64 result only.
    //
    // Prior pins (kept for audit trail):
    //   - 2026-05-14 (pre-#691):   35fd85a47bab905d88c1fa96d8910e02d1248cbd17e9527809f5c4f9559c1f3e (47118 bytes, LEVEL_3_0 hardcoded)
    //   - 2026-05-23 post-#691:    8ba4002b2ecec7662e1e1e04f24cb9720dafdb63354bc54099794220bd793ddf (47118 bytes, dynamic LEVEL_2_1, PRO_MAIN)
    //   - 2026-05-23 post-V0.4.C:  c3a4d284e1b51741d0301ce469ffd9d0a886cae3885e1066c1f3049b555897b5 (47119 bytes, PRO_HIGH, set1=0)
    //   - 2026-06 Phase 6 (cf v2): 597e1db6e403e98454a9218bae3c8dc4178581c0302677668cb1a94dd618236d (47128 bytes, 4-domain streaming, chunk_frame v2)
    //
    // If this assert ever fires:
    //   1. Print both hashes so the diff is captured in the log.
    //   2. Check whether the encoder source has intentionally
    //      changed (new OH264 fork SHA, new STC parameters, new
    //      stego pipeline behaviour, shim level/qp/refs/profile/POC
    //      change). If yes, re-record the hash.
    //   3. If no intentional change, the encoder has become non-
    //      deterministic on this arch — investigate before merging.
    // Re-baselined 2026-06-11 for chunk_frame WIRE FORMAT v3 (#839). The
    // v2-era Phase-6 pin (597e1db6…, in the audit trail above) went stale the
    // moment #839 migrated the streaming session to chunk_frame v3 — the v3
    // header layout changes the embedded-message bytes (SAME total length,
    // 47128 B; different content). `git bisect` localized the change to the
    // #839 cluster (1f20526c #839 + merge 30e2d835); the J-UNIWARD FP cost
    // path is untouched, so the later perf work (parallel cost/MVD loops,
    // delta-field memoize, wavelet SIMD #848) is bit-exact and NOT the cause —
    // the output was already this hash before all of them. Pinned from macOS
    // aarch64; CI MUST STILL confirm the Linux x86_64 + aarch64 hashes match it
    // (the cross-arch determinism claim, #835) per the workflow_dispatch
    // re-pin protocol documented above.
    const PINNED_SHA256: &str =
        "54c11576e2dd28607e84a20303388aeef8c4d46b141b021168d3d864c6aea911";
    const PINNED_LEN: usize = 47128;
    assert_eq!(
        stego.len(), PINNED_LEN,
        "stego length {} on {os}-{arch} differs from pinned {}",
        stego.len(), PINNED_LEN,
    );
    assert_eq!(
        hash_hex, PINNED_SHA256,
        "cross-arch determinism regression on {os}-{arch}:\n  \
         current:  {hash_hex}\n  pinned:   {PINNED_SHA256}",
    );
}
