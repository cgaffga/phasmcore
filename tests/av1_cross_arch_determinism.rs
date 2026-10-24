// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// AV1 assembly + cross-arch determinism gate (2026-10-24).
//
// phasm-core builds rav1e with its `asm` feature (NEON on aarch64,
// SSE/AVX2/AVX-512 on x86_64) and dav1d with its assembly. This test
// checks that neither changes what phasm writes:
//
//   1. Every rav1e CPU level this machine supports (RUST up to NEON, or
//      RUST up to AVX512ICL) produces the same stego stream as the
//      pure-Rust kernels — `RAV1E_CPU_TARGET` (rav1e) and `CPU_TARGET`
//      (its scene-cut detector, av-scenechange) force the level per encode.
//   2. dav1d forced to its C paths (`dav1d_set_cpu_flags_mask(0)`)
//      produces the same stego stream as dav1d with its assembly — Pass 2
//      rebuilds every GOP from a dav1d decode of its Pass-1 packets.
//   3. Every one of those streams decodes back to the message.
//   4. The stream matches the pinned hash, which was recorded on macOS
//      aarch64 — so x86_64 and Linux produce the same bytes as well.
//
// 1–3 are the assembly checks and hold on any machine on their own; 4 is
// the cross-arch claim. Hashes for every level are printed before
// anything is asserted, so one CI log shows the whole picture.
//
// Runs in CI via `.github/workflows/av1-cross-arch.yml` (manual trigger,
// no artifacts — see the policy block there). The test prints text only:
// target, level, sizes and hex hashes, never stream bytes.
//
// Run locally:
//   cargo test --release --features av1-encoder \
//     --test av1_cross_arch_determinism -- --nocapture

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use phasm_core::Av1StreamingEncodeParams;
use phasm_core::codec::av1::stego::session::Av1StreamingDecodeSession;
use phasm_core::codec::av1::stego::whole_video::{
    Av1SliceYuvSource, Pass1CacheBudget, av1_encode_streaming,
};
use phasm_core::stego::{crypto, frame, payload};
use phasm_rav1e::config::CpuFeatureLevel;
use sha2::{Digest, Sha256};

const MESSAGE: &str = "phasm av1 cross-arch determinism v1";
const PASSPHRASE: &str = "av1-cross-arch-pass";

/// rav1e CPU levels from plain Rust up, as `RAV1E_CPU_TARGET` names them.
#[cfg(target_arch = "x86_64")]
const RAV1E_LEVELS: &[&str] = &["RUST", "SSE2", "SSSE3", "SSE4_1", "AVX2", "AVX512", "AVX512ICL"];
#[cfg(target_arch = "aarch64")]
const RAV1E_LEVELS: &[&str] = &["RUST", "NEON"];
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const RAV1E_LEVELS: &[&str] = &["RUST"];

struct Fixture {
    name: &'static str,
    w: u32,
    h: u32,
    n_frames: u32,
    gop: u32,
    /// SHA-256 of the stego stream, recorded on macOS aarch64 on
    /// 2026-10-24 (phasm-rav1e `abe19cd8`, phasm-dav1d `acadfd78`), where
    /// RUST and NEON, dav1d asm and C all produced it. Any change to what
    /// the encoder writes (fork pins, cost model, framing) moves it:
    /// re-record and say why here.
    pinned_sha256: &'static str,
    pinned_len: usize,
}

const FIXTURES: &[Fixture] = &[
    // One GOP.
    Fixture {
        name: "256x144x4",
        w: 256,
        h: 144,
        n_frames: 4,
        gop: 4,
        pinned_sha256: "bad838c2898d764338defe15b629aa12ef8fefbe21e3cb92724a3ee710d92ae1",
        pinned_len: 81358,
    },
    // Two GOPs, frame size not a multiple of 8 (padded edge blocks).
    Fixture {
        name: "250x142x6",
        w: 250,
        h: 142,
        n_frames: 6,
        gop: 3,
        pinned_sha256: "296d14cbeb8b3da6323b59bf78d5c3f317e14650658453ae5015e16e796feb1d",
        pinned_len: 122462,
    },
    // 64x64 superblocks and the large transforms.
    Fixture {
        name: "640x360x4",
        w: 640,
        h: 360,
        n_frames: 4,
        gop: 4,
        pinned_sha256: "0fbf65e8bf1c6165ac568c6a5f9e71d50076d8b2fcebd897bb5383d53dbef910",
        pinned_len: 407620,
    },
];

/// Deterministic I420 clip with motion: a noise texture pans diagonally
/// under a smooth gradient band, and a flat square crosses it the other
/// way. Intra and inter prediction, motion search and the transform sizes
/// all get real work; flat grey would starve the cover.
fn synthetic_clip(w: u32, h: u32, n: u32) -> Vec<u8> {
    let (w, h, n) = (w as usize, h as usize, n as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let frame = w * h + 2 * cw * ch;
    // Texture big enough for the pan.
    let (tw, th) = (w + 2 * n + 2, h + n + 2);
    let mut s: u32 = 0x2545_F491;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let luma_tex: Vec<u8> = (0..tw * th).map(|_| (next() >> 24) as u8).collect();
    let chroma_tex: Vec<u8> = (0..tw * th).map(|_| 112 + (next() >> 27) as u8).collect();

    let mut out = vec![0u8; frame * n];
    for f in 0..n {
        let base = f * frame;
        let (sq_x, sq_y, sq) = (w.saturating_sub(48 + 6 * f), h / 3, 32);
        for y in 0..h {
            for x in 0..w {
                let v = if y < h / 4 {
                    ((x * 255) / w.max(1)) as u8 ^ ((y & 3) as u8)
                } else if (sq_x..sq_x + sq).contains(&x) && (sq_y..sq_y + sq).contains(&y) {
                    200
                } else {
                    luma_tex[(y + f) * tw + x + 2 * f]
                };
                out[base + y * w + x] = v;
            }
        }
        for plane in 0..2 {
            let off = base + w * h + plane * cw * ch;
            for y in 0..ch {
                for x in 0..cw {
                    out[off + y * cw + x] = chroma_tex[(y + plane) * tw + x + f];
                }
            }
        }
    }
    out
}

/// `payload::encode_payload` → encrypt → `frame::build_frame`, as the CLI
/// and both bridges frame the primary.
fn framed_primary() -> Vec<u8> {
    let p = payload::encode_payload(MESSAGE, &[]).expect("encode payload");
    let (ct, nonce, salt) = crypto::encrypt(&p, PASSPHRASE).expect("encrypt");
    frame::build_frame(p.len(), &salt, &nonce, &ct)
}

fn encode(fx: &Fixture, yuv: &[u8], framed: &[u8]) -> Vec<u8> {
    let params = Av1StreamingEncodeParams {
        width: fx.w,
        height: fx.h,
        quantizer: 60,
        gop_size: fx.gop,
        total_frames_hint: fx.n_frames,
    };
    let mut src = Av1SliceYuvSource::new(yuv, fx.w, fx.h, fx.n_frames, fx.gop);
    av1_encode_streaming(
        &mut src,
        fx.n_frames,
        params,
        framed,
        PASSPHRASE,
        Pass1CacheBudget::from_env_or_default(),
        None,
    )
    .unwrap_or_else(|e| panic!("{}: stego encode failed: {e:?}", fx.name))
}

fn decodes_to_message(stego: &[u8]) -> bool {
    let mut dec = Av1StreamingDecodeSession::create(PASSPHRASE);
    dec.push_bytes(stego);
    matches!(dec.finish_primary_payload(), Ok(p) if p.text == MESSAGE)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn set_env(key: &str, value: &str) {
    // SAFETY: the only test in this binary; nothing else reads the
    // environment concurrently.
    unsafe { std::env::set_var(key, value) }
}

fn set_dav1d_asm(enabled: bool) {
    // SAFETY: plain global store in dav1d; no decoder is open here.
    unsafe { core_dav1d_sys::dav1d_set_cpu_flags_mask(if enabled { !0 } else { 0 }) }
}

struct Run {
    fixture: &'static str,
    rav1e: String,
    dav1d: &'static str,
    len: usize,
    sha256: String,
    decoded: bool,
}

#[test]
fn av1_asm_and_cross_arch_determinism() {
    // Fixed salt + nonce (`stego/crypto.rs`); without it every encode of
    // the same input differs, by design.
    set_env("PHASM_DETERMINISTIC_SEED", "42");
    let framed = framed_primary();
    let target = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);

    // rav1e levels this machine runs: a level above the detected one
    // falls back to the detected one, so keep each effective level once.
    let mut levels: Vec<String> = Vec::new();
    for name in RAV1E_LEVELS {
        set_env("RAV1E_CPU_TARGET", name);
        let effective = format!("{:?}", CpuFeatureLevel::default());
        if !levels.contains(&effective) {
            levels.push(effective);
        }
    }
    assert_eq!(levels[0], "RUST", "RAV1E_CPU_TARGET=RUST was not honoured");
    eprintln!("target={target} rav1e_levels={levels:?}");

    let mut runs: Vec<Run> = Vec::new();
    for fx in FIXTURES {
        let yuv = synthetic_clip(fx.w, fx.h, fx.n_frames);
        let mut record = |rav1e: &str, dav1d_asm: bool| {
            set_env("RAV1E_CPU_TARGET", rav1e);
            // rav1e's scene-cut detector, av-scenechange, has assembly of
            // its own (on for every target in the fork's manifest) and
            // reads its level from `CPU_TARGET`; same level names.
            set_env("CPU_TARGET", rav1e);
            set_dav1d_asm(dav1d_asm);
            let stego = encode(fx, &yuv, &framed);
            let decoded = decodes_to_message(&stego);
            let run = Run {
                fixture: fx.name,
                rav1e: rav1e.to_string(),
                dav1d: if dav1d_asm { "asm" } else { "c" },
                len: stego.len(),
                sha256: sha256_hex(&stego),
                decoded,
            };
            // Text only — never stream bytes.
            eprintln!(
                "target={target} fixture={} rav1e={} dav1d={} len={} sha256={} decoded={}",
                run.fixture, run.rav1e, run.dav1d, run.len, run.sha256, run.decoded
            );
            runs.push(run);
        };
        for level in &levels {
            record(level, true);
        }
        record("RUST", false);
    }
    set_dav1d_asm(true);

    for run in &runs {
        assert!(
            run.decoded,
            "{} rav1e={} dav1d={}: stream did not decode to the message",
            run.fixture, run.rav1e, run.dav1d
        );
    }
    for fx in FIXTURES {
        let mut of_fx = runs.iter().filter(|r| r.fixture == fx.name);
        let reference = of_fx.next().expect("RUST run");
        for run in of_fx {
            assert_eq!(
                (run.len, &run.sha256),
                (reference.len, &reference.sha256),
                "{} on {target}: rav1e={} dav1d={} differs from rav1e=RUST dav1d=asm",
                fx.name,
                run.rav1e,
                run.dav1d
            );
        }
        assert_eq!(
            (reference.len, reference.sha256.as_str()),
            (fx.pinned_len, fx.pinned_sha256),
            "{} on {target}: differs from the pinned macOS aarch64 stream",
            fx.name
        );
    }
}
