// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! No-shadow round-trip via the AV1 streaming pull API
//! (`av1_encode_streaming` + `Av1StreamingDecodeSession`).
//!
//! Production gate. Pull-API correctness coverage that replaced the
//! deleted byte-identity gate (#219) and the deleted
//! `av1_session_multi_frame_gop` push-API test when the push API was
//! removed (#225/#187, 2026-06-29).
//!
//! Encodes a real-content YUV through the production pull entry, then
//! decodes the resulting AV1 OBU stream and asserts the recovered
//! payload matches the original. Covers:
//!   - `Av1SliceYuvSource` rewind + per-GOP pull contract
//!   - `av1_encode_streaming` two-pass encode + per-GOP proportional
//!     primary plan (2026-10-23 — was an even split)
//!   - Per-GOP `chunk_frame_v3.1` framing
//!   - `Av1StreamingDecodeSession` OBU-walker accumulator + decrypt

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

mod common;

use phasm_core::codec::av1::stego::primary_plan::gop_primary_caps;
use phasm_core::codec::av1::stego::session::Av1StreamingDecodeSession;
use phasm_core::codec::av1::stego::whole_video::{
    av1_encode_streaming, Av1GopYuvSource, Av1SliceYuvSource, Pass1CacheBudget,
};
use phasm_core::stego::{crypto, frame, payload};
use phasm_core::{av1_probe_gops_at, Av1StreamingEncodeParams};

/// Pull a corpus YUV from the test-vector clip via ffmpeg. Real
/// content so rav1e's RDO has actual signal to encode — synthetic
/// patterns at small fixtures starve the cover and the encode trips
/// `FrameCorrupted` before the assert runs. Same fixture the
/// byte-identity gate used pre-deletion.
fn corpus_yuv_concat(w: u32, h: u32, n: u32) -> Vec<u8> {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source/IMG_4138.MOV");
    assert!(p.exists(), "corpus fixture missing: {}", p.display());
    let vf = format!("scale={w}:{h}:force_original_aspect_ratio=disable");
    let out = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&p)
        .args([
            "-frames:v",
            &n.to_string(),
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
    assert_eq!(
        out.stdout.len(),
        (w * h * 3 / 2 * n) as usize,
        "ffmpeg produced unexpected byte count"
    );
    out.stdout
}

fn params(w: u32, h: u32, gop: u32, n: u32, quantizer: usize) -> Av1StreamingEncodeParams {
    Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer,
        gop_size: gop,
        total_frames_hint: n,
    }
}

/// `payload::encode_payload` → encrypt → `frame::build_frame`, as the CLI
/// and both bridges frame the primary.
fn framed_primary(text: &str, pass: &str) -> Vec<u8> {
    let p = payload::encode_payload(text, &[]).expect("encode primary payload");
    let (ct, nonce, salt) = crypto::encrypt(&p, pass).expect("encrypt primary");
    frame::build_frame(p.len(), &salt, &nonce, &ct)
}

/// Deterministic high-entropy text whose framed primary is as close to
/// `target` bytes as possible without exceeding it. Sized from
/// `encode_payload` + the frame overhead, so no Argon2 per probe.
fn text_for_framed_len(target: usize) -> String {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let pool: String = (0..target * 2 + 64)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (33 + (s % 94) as u8) as char
        })
        .collect();
    let framed_len = |n: usize| {
        let p = payload::encode_payload(&pool[..n], &[]).unwrap().len();
        p + if p > u16::MAX as usize {
            frame::FRAME_OVERHEAD_EXT
        } else {
            frame::FRAME_OVERHEAD
        }
    };
    let (mut lo, mut hi) = (0usize, pool.len());
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if framed_len(mid) <= target {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    pool[..lo].to_string()
}

fn encode_decode(
    yuv: &[u8],
    p: Av1StreamingEncodeParams,
    text: &str,
    pass: &str,
    budget: Pass1CacheBudget,
) -> String {
    let framed = framed_primary(text, pass);
    let n = p.total_frames_hint;
    let mut src = Av1SliceYuvSource::new(yuv, p.width, p.height, n, p.gop_size);
    let stego = av1_encode_streaming(&mut src, n, p, &framed, pass, budget, None)
        .expect("streaming encode");
    assert!(!stego.is_empty(), "encoder produced no bytes");

    let mut dec = Av1StreamingDecodeSession::create(pass);
    dec.push_bytes(&stego);
    let recovered = dec
        .finish_primary_payload()
        .expect("decode finish_primary_payload");
    assert!(
        recovered.files.is_empty(),
        "no files were encoded — recovered.files must be empty, got {} entry/ies",
        recovered.files.len()
    );
    recovered.text
}

fn roundtrip_at_fixture(
    w: u32,
    h: u32,
    gop: u32,
    n: u32,
    quantizer: usize,
    primary_text: &str,
    primary_pass: &str,
) {
    let yuv = corpus_yuv_concat(w, h, n);
    let got = encode_decode(
        &yuv,
        params(w, h, gop, n, quantizer),
        primary_text,
        primary_pass,
        Pass1CacheBudget::from_env_or_default(),
    );
    assert_eq!(got, primary_text, "recovered primary text mismatch");
}

/// 3 full GOPs, no partial trailing — exercises the steady-state
/// per-GOP encode + chunk-split path.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull ~30-60 s; run with \
            --ignored --test-threads=1"]
fn pull_no_shadow_roundtrip_3_full_gops() {
    roundtrip_at_fixture(
        256, 144, 4, 12, // w, h, gop, n_frames
        30,              // quantizer (high quality, lots of cover)
        "pull-API no-shadow round-trip — substantive primary message",
        "pull-no-shadow-pass",
    );
}

/// A primary frame over the video bound is refused up front with
/// `MessageTooLarge` — before any GOP is pulled or encoded — rather than
/// encoding a clip no decoder will ever read back.
#[test]
fn pull_no_shadow_refuses_frame_over_video_bound() {
    use phasm_core::codec::av1::stego::orchestrator::Av1StegoError;
    use phasm_core::stego::error::StegoError;

    let (w, h, gop, n) = (64u32, 64u32, 4u32, 4u32);
    let yuv = vec![128u8; (w * h * 3 / 2 * n) as usize];
    let oversized = vec![0u8; frame::VIDEO_MAX_FRAME_BYTES + 1];
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let res = av1_encode_streaming(
        &mut src,
        n,
        params(w, h, gop, n, 100),
        &oversized,
        "pass",
        Pass1CacheBudget::from_env_or_default(),
        None,
    );
    assert!(
        matches!(res, Err(Av1StegoError::Stego(StegoError::MessageTooLarge))),
        "expected MessageTooLarge, got {:?}",
        res.map(|v| v.len())
    );
}

/// 2 full GOPs + 1 partial 3-frame trailing GOP — validates the
/// trailing-partial-GOP path inside `av1_encode_streaming`.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull ~30-60 s; run with \
            --ignored --test-threads=1"]
fn pull_no_shadow_roundtrip_with_partial_tail() {
    roundtrip_at_fixture(
        256, 144, 4, 11, // n_frames not divisible by gop
        30,
        "pull-API no-shadow with partial-tail GOP",
        "pull-no-shadow-partial",
    );
}

/// The capacity gate for the proportional plan (D2's two-gate pattern in
/// one test): a primary the even split provably cannot carry — it exceeds
/// `n_gops × min(GOP capacity)` because of a 1-frame trailing GOP — must
/// round-trip, which also proves the decoder reads unequal chunks.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn pull_no_shadow_roundtrip_beyond_even_split() {
    let (w, h, gop, n) = (256u32, 144u32, 8u32, 17u32); // GOPs of 8, 8, 1 frames
    let p = params(w, h, gop, n, 30);
    let yuv = corpus_yuv_concat(w, h, n);

    let mut probe_src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    probe_src.rewind().expect("rewind");
    let cover: Vec<usize> = av1_probe_gops_at(&mut probe_src, n, p, &[0, 1, 2])
        .expect("probe")
        .into_iter()
        .map(|(_, cb)| cb)
        .collect();
    let caps = gop_primary_caps(&cover, &[0; 3]);
    let even_ceiling = 3 * caps.iter().map(|c| c.payload_bytes).min().unwrap();
    let sigma: usize = caps.iter().map(|c| c.payload_bytes).sum();

    let text = text_for_framed_len((even_ceiling + sigma) / 2);
    let framed_len = framed_primary(&text, "beyond-even").len();
    assert!(
        framed_len > even_ceiling,
        "precondition: framed {framed_len} must exceed the even split's ceiling \
         {even_ceiling} (cover {cover:?})"
    );

    let got = encode_decode(&yuv, p, &text, "beyond-even", Pass1CacheBudget::from_env_or_default());
    assert_eq!(got, text);
}

/// Cache disabled (RAM 0, disk 0): every GOP is recomputed in Pass 2,
/// which must be byte-identical to Pass 1 or the plan and the embed
/// disagree.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn pull_no_shadow_roundtrip_cache_disabled() {
    let (w, h, gop, n) = (256u32, 144u32, 4u32, 12u32);
    let yuv = corpus_yuv_concat(w, h, n);
    let text = "every GOP recomputed in Pass 2";
    let budget = Pass1CacheBudget::new(0, 0, common::scratch_dir().into());
    let got = encode_decode(&yuv, params(w, h, gop, n, 30), text, "no-cache", budget);
    assert_eq!(got, text);
}

/// A message far smaller than the clip: small per-GOP chunks.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn pull_no_shadow_roundtrip_tiny_message() {
    let (w, h, gop, n) = (256u32, 144u32, 4u32, 12u32);
    let yuv = corpus_yuv_concat(w, h, n);
    let got = encode_decode(
        &yuv,
        params(w, h, gop, n, 30),
        "hi!",
        "tiny",
        Pass1CacheBudget::from_env_or_default(),
    );
    assert_eq!(got, "hi!");
}

/// A 1-frame trailing GOP goes through the multi-frame Pass 1 path (the
/// legacy single-frame primitive is no longer used for it).
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn pull_no_shadow_roundtrip_one_frame_tail() {
    let (w, h, gop, n) = (256u32, 144u32, 4u32, 9u32); // GOPs of 4, 4, 1 frames
    let yuv = corpus_yuv_concat(w, h, n);
    let text = "a one-frame trailing GOP";
    let got = encode_decode(
        &yuv,
        params(w, h, gop, n, 30),
        text,
        "one-frame-tail",
        Pass1CacheBudget::from_env_or_default(),
    );
    assert_eq!(got, text);
}

/// A flat grey GOP in the middle cannot carry even a chunk header, so the
/// whole message must land before it and the GOPs from it on stay natural
/// — the decoder stops once the message is complete.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn pull_no_shadow_roundtrip_window_ends_at_flat_gop() {
    let (w, h, gop, n) = (256u32, 144u32, 1u32, 3u32); // one frame per GOP
    let mut yuv = corpus_yuv_concat(w, h, n);
    let frame_size = (w * h * 3 / 2) as usize;
    yuv[frame_size..2 * frame_size].fill(128); // GOP 1: flat grey
    let p = params(w, h, gop, n, 30);

    let mut probe_src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    probe_src.rewind().expect("rewind");
    let cover: Vec<usize> = av1_probe_gops_at(&mut probe_src, n, p, &[0, 1, 2])
        .expect("probe")
        .into_iter()
        .map(|(_, cb)| cb)
        .collect();
    let caps = gop_primary_caps(&cover, &[0; 3]);
    assert_eq!(
        phasm_core::codec::av1::stego::primary_plan::window_end(&caps),
        1,
        "precondition: the grey GOP must be header-incapable (cover {cover:?})"
    );

    let text = "lands entirely in GOP 0";
    let got = encode_decode(&yuv, p, text, "flat-gop", Pass1CacheBudget::from_env_or_default());
    assert_eq!(got, text);
}

/// The progress bar follows the work, not a fixed pass split: Pass 2 is
/// cheap while the Pass-1 cache holds every GOP and as long as Pass 1 when it
/// holds none (every GOP natural-encoded again). Events never go back, end at
/// exactly 1.0, and say which pass and GOP they belong to.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn pull_no_shadow_progress_tracks_work() {
    use phasm_core::codec::av1::stego::encode_progress::{Av1EncodePhase, Av1EncodeProgress};

    let (w, h, gop, n) = (256u32, 144u32, 4u32, 12u32); // 3 GOPs
    let yuv = corpus_yuv_concat(w, h, n);
    let p = params(w, h, gop, n, 30);
    let framed = framed_primary("progress follows the work", "progress-pass");

    for (budget, cached) in [
        (Pass1CacheBudget::from_env_or_default(), true),
        (Pass1CacheBudget::new(0, 0, common::scratch_dir().into()), false),
    ] {
        let mut events: Vec<Av1EncodeProgress> = Vec::new();
        {
            let mut on_progress = |e: Av1EncodeProgress| events.push(e);
            let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
            av1_encode_streaming(&mut src, n, p, &framed, "progress-pass", budget, Some(&mut on_progress))
                .expect("encode");
        }
        assert!(
            events.windows(2).all(|e| e[1].fraction >= e[0].fraction),
            "fraction went back: {events:?}"
        );
        assert_eq!(events.last().unwrap().fraction, 1.0);
        assert!(events.iter().take(events.len() - 1).all(|e| e.fraction < 1.0));

        let encode: Vec<_> = events.iter().filter(|e| e.phase == Av1EncodePhase::Encode).collect();
        assert_eq!(encode.iter().map(|e| e.gop).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert!(encode.iter().all(|e| e.n_gops == 3));
        assert!(events.iter().filter(|e| e.phase == Av1EncodePhase::Embed).count() >= 3);

        let end_of_pass1 = encode.last().unwrap().fraction;
        if cached {
            assert!(end_of_pass1 > 0.9, "cached Pass 2 is cheap: Pass 1 ends at {end_of_pass1}");
        } else {
            assert!(end_of_pass1 < 0.6, "uncached Pass 2 re-encodes: Pass 1 ends at {end_of_pass1}");
        }
    }
}

/// SHA-256 and length of the stego stream [`pull_no_shadow_aligned_output_is_unchanged`]
/// encodes, captured 2026-10-24 before the Pass-2 rebuild (option 3) landed.
const GOLDEN_256X144: &str = "33aa313f28d12504318e6fd7a58d38da92e5a0e4df50b7c5c049e523d36af3aa";
const GOLDEN_256X144_LEN: usize = 95_837;

/// At a frame size whose width and height are multiples of 8, the encode's
/// stego output stays byte-identical across the Pass-2 rebuild: rebuilding a
/// GOP from its natural packets must hand Pass 2 exactly what the encoder
/// recorded, and the padding fix has nothing to replicate at aligned sizes.
///
/// The golden belongs to this machine's ffmpeg scaler and rav1e build (and
/// to aarch64 — the cost math is not bit-identical across architectures).
/// The framed primary is opaque bytes, not an encrypted payload, so the run
/// is deterministic; nothing decodes it.
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore = "AV1 encode + ffmpeg corpus; --ignored --test-threads=1"]
fn pull_no_shadow_aligned_output_is_unchanged() {
    use sha2::{Digest, Sha256};

    let (w, h, gop, n) = (256u32, 144u32, 4u32, 12u32);
    let yuv = corpus_yuv_concat(w, h, n);
    let framed: Vec<u8> = (0..300u32).map(|i| (i * 131 + 7) as u8).collect();
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let output = av1_encode_streaming(
        &mut src,
        n,
        params(w, h, gop, n, 30),
        &framed,
        "golden-pass",
        Pass1CacheBudget::from_env_or_default(),
        None,
    )
    .expect("encode");
    let hash: String = Sha256::digest(&output).iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        (hash.as_str(), output.len()),
        (GOLDEN_256X144, GOLDEN_256X144_LEN),
        "stego output changed: sha256 {hash}, {} B",
        output.len()
    );
}

/// Encode `framed` twice — every GOP rebuilt from its cached packets
/// (default budget) and every GOP re-encoded (budget 0 / 0) — and require
/// the same stego bytes, which decode to `text`.
fn assert_rebuild_and_reencode_identical(w: u32, h: u32) {
    let (gop, n) = (4u32, 12u32);
    let yuv = corpus_yuv_concat(w, h, n);
    let p = params(w, h, gop, n, 30);
    let (text, pass) = ("rebuilt and re-encoded GOPs embed the same bytes", "identical-paths");
    let framed = framed_primary(text, pass);
    let encode = |budget: Pass1CacheBudget| {
        let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
        av1_encode_streaming(&mut src, n, p, &framed, pass, budget, None).expect("encode")
    };
    let rebuilt = encode(Pass1CacheBudget::from_env_or_default());
    let reencoded = encode(Pass1CacheBudget::new(0, 0, common::scratch_dir().into()));
    assert!(rebuilt == reencoded, "{w}x{h}: rebuild and re-encode paths emit different bytes");
    let mut dec = Av1StreamingDecodeSession::create(pass);
    dec.push_bytes(&rebuilt);
    assert_eq!(dec.finish_primary_payload().expect("decode").text, text);
}

/// Pass 2 rebuilds a cached GOP from its natural packets with a dav1d
/// decode; an uncached one is re-encoded. Both must hand the embed the
/// same data.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn pull_no_shadow_rebuild_and_reencode_emit_identical_bytes_aligned() {
    assert_rebuild_and_reencode_identical(256, 144);
}

/// 250×142: rav1e pads to 256×144, and its padding pixels differ from a
/// decoder's — the cost input must not see them.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn pull_no_shadow_rebuild_and_reencode_emit_identical_bytes_unaligned() {
    assert_rebuild_and_reencode_identical(250, 142);
}

/// A source that logs every call: `None` for a rewind, `Some(g)` for a GOP.
struct LoggingSource<'a> {
    inner: Av1SliceYuvSource<'a>,
    log: Vec<Option<u32>>,
}

impl Av1GopYuvSource for LoggingSource<'_> {
    fn gop_yuv(
        &mut self,
        gop_index: u32,
    ) -> Result<Vec<u8>, phasm_core::codec::av1::stego::orchestrator::Av1StegoError> {
        self.log.push(Some(gop_index));
        self.inner.gop_yuv(gop_index)
    }
    fn rewind(&mut self) -> Result<(), phasm_core::codec::av1::stego::orchestrator::Av1StegoError> {
        self.log.push(None);
        self.inner.rewind()
    }
}

/// With every GOP cached, Pass 2 rebuilds from packets and never touches
/// the source again — on a phone that is a second full decode of the input
/// video saved. With nothing cached it re-reads the source once, in order.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn pull_no_shadow_pass2_leaves_source_alone_when_cached() {
    let (w, h, gop, n) = (256u32, 144u32, 4u32, 12u32); // 3 GOPs
    let yuv = corpus_yuv_concat(w, h, n);
    let p = params(w, h, gop, n, 30);
    let framed = framed_primary("source pulls", "pulls");
    let one_sweep: Vec<Option<u32>> =
        std::iter::once(None).chain((0..3).map(Some)).collect();
    for (budget, sweeps) in [
        (Pass1CacheBudget::from_env_or_default(), 1),
        (Pass1CacheBudget::new(0, 0, common::scratch_dir().into()), 2),
    ] {
        let mut src = LoggingSource { inner: Av1SliceYuvSource::new(&yuv, w, h, n, gop), log: Vec::new() };
        av1_encode_streaming(&mut src, n, p, &framed, "pulls", budget, None).expect("encode");
        assert_eq!(src.log, one_sweep.repeat(sweeps), "{sweeps} source sweep(s) expected");
    }
}

/// Pass 2's cost relative to Pass 1 when the cache holds every GOP: each
/// GOP is rebuilt from its natural packets (dav1d decode + harvest) and
/// embedded (STC + replay). Sets `encode_progress::EMBED_WEIGHT`, and pins
/// the point of the rebuild — the re-encode it replaced made this ~1.0.
/// 1080p q100, 2 GOPs of 30, a ~6 KB primary (~half of the ~12.5 KB the
/// stratified probe advertises for this clip).
#[test]
#[ignore = "timing: 1080p AV1 encode ~2 min; run alone with --ignored --nocapture"]
fn pull_no_shadow_pass2_cost_ratio() {
    use phasm_core::codec::av1::stego::encode_progress::{Av1EncodePhase, Av1EncodeProgress};
    use std::time::Instant;

    let (w, h, gop, n) = (1920u32, 1080u32, 30u32, 60u32);
    let yuv = corpus_yuv_concat(w, h, n);
    let p = params(w, h, gop, n, 100);
    let text = text_for_framed_len(6000);
    let framed = framed_primary(&text, "ratio");
    let (mut last_encode, mut last_embed) = (None, None);
    let t0 = Instant::now();
    {
        let mut on_progress = |e: Av1EncodeProgress| match e.phase {
            Av1EncodePhase::Encode => last_encode = Some(Instant::now()),
            Av1EncodePhase::Embed => last_embed = Some(Instant::now()),
        };
        let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
        av1_encode_streaming(
            &mut src,
            n,
            p,
            &framed,
            "ratio",
            Pass1CacheBudget::from_env_or_default(),
            Some(&mut on_progress),
        )
        .expect("encode");
    }
    let (enc, emb) = (last_encode.unwrap(), last_embed.unwrap());
    let pass1 = (enc - t0).as_secs_f64();
    let pass2 = (emb - enc).as_secs_f64();
    let ratio = pass2 / pass1;
    eprintln!("[pass2_cost_ratio] pass1 {pass1:.2}s, pass2 {pass2:.2}s, pass2/pass1 {ratio:.4}");
    assert!(ratio < 0.15, "Pass 2 is {ratio:.3} of Pass 1 — the rebuild should make it cheap");
}

/// A cache miss after hits, end to end: every GOP spills to disk, and GOP
/// 1's spill is deleted between the passes. Pass 2 rebuilds GOP 0 from its
/// packets, re-encodes GOP 1 from the source — rewinding and pulling 0, 1 in
/// order, as the source contract requires — and rebuilds GOP 2. The bytes
/// must match an all-cached encode, and the progress bar, which did not
/// expect the miss, must still only move forward and end at 1.0.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn pull_no_shadow_missing_spill_reencodes_in_order() {
    use phasm_core::codec::av1::stego::encode_progress::{Av1EncodePhase, Av1EncodeProgress};

    let (w, h, gop, n) = (256u32, 144u32, 4u32, 12u32); // 3 GOPs
    let yuv = corpus_yuv_concat(w, h, n);
    let p = params(w, h, gop, n, 30);
    let (text, pass) = ("a spill that cannot be read back is re-encoded", "missing-spill");
    let framed = framed_primary(text, pass);
    let reference = {
        let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
        av1_encode_streaming(&mut src, n, p, &framed, pass, Pass1CacheBudget::from_env_or_default(), None)
            .expect("reference encode")
    };

    let parent = std::path::PathBuf::from(common::scratch_dir())
        .join(format!("av1-missing-spill-{}", std::process::id()));
    std::fs::create_dir_all(&parent).expect("scratch parent");
    let delete_spill = |gop_idx: u32| -> bool {
        let name = format!("gop_{gop_idx:08}.bin");
        std::fs::read_dir(&parent)
            .into_iter()
            .flatten()
            .flatten()
            .map(|session| session.path().join(&name))
            .any(|spill| std::fs::remove_file(spill).is_ok())
    };
    let mut deleted = false;
    let mut fractions: Vec<f32> = Vec::new();
    let mut src = LoggingSource { inner: Av1SliceYuvSource::new(&yuv, w, h, n, gop), log: Vec::new() };
    let output = {
        let mut on_progress = |e: Av1EncodeProgress| {
            fractions.push(e.fraction);
            if e.phase == Av1EncodePhase::Encode && e.gop == e.n_gops {
                deleted = delete_spill(1);
            }
        };
        av1_encode_streaming(
            &mut src,
            n,
            p,
            &framed,
            pass,
            Pass1CacheBudget::new(0, 1 << 30, parent.clone()),
            Some(&mut on_progress),
        )
        .expect("encode")
    };
    let _ = std::fs::remove_dir_all(&parent);

    assert!(deleted, "GOP 1 was not spilled to disk");
    assert!(output == reference, "a re-encoded GOP among rebuilt ones changed the bytes");
    assert_eq!(
        src.log,
        vec![None, Some(0), Some(1), Some(2), None, Some(0), Some(1)],
        "Pass 2 must rewind and pull forward to the missing GOP only"
    );
    assert!(fractions.windows(2).all(|f| f[1] >= f[0]), "progress went back: {fractions:?}");
    assert_eq!(fractions.last().copied(), Some(1.0));
    let mut dec = Av1StreamingDecodeSession::create(pass);
    dec.push_bytes(&output);
    assert_eq!(dec.finish_primary_payload().expect("decode").text, text);
}
