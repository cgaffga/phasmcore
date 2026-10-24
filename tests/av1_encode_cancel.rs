// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! A cancel during the AV1 encode must surface as `Cancelled`.
//!
//! `stc_embed` returns `None` when the global cancel flag is set — the same
//! `None` an infeasible chunk produces. Pass 2 shrinks an infeasible chunk
//! and re-plans with what it learned, so without an explicit cancel check a
//! user cancel was "learned" as GOP capacity and reported as
//! `MessageTooLarge`.
//!
//! Own test binary: the cancel flag is process-global and would cancel any
//! test running alongside it.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

mod common;

use phasm_core::codec::av1::stego::orchestrator::Av1StegoError;
use phasm_core::codec::av1::stego::whole_video::{
    av1_encode_streaming, av1_encode_with_shadows_streaming, Av1GopYuvSource, Av1SliceYuvSource,
    Pass1CacheBudget,
};
use phasm_core::stego::error::StegoError;
use phasm_core::stego::{crypto, frame, payload, progress};
use phasm_core::Av1StreamingEncodeParams;

fn corpus_yuv_concat(w: u32, h: u32, n: u32) -> Vec<u8> {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source/IMG_4138.MOV");
    assert!(p.exists(), "corpus fixture missing: {}", p.display());
    let vf = format!("scale={w}:{h}:force_original_aspect_ratio=disable");
    let out = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&p)
        .args(["-frames:v", &n.to_string(), "-vf", &vf])
        .args(["-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        .output()
        .expect("ffmpeg launch");
    assert!(out.status.success(), "ffmpeg yuv extract failed");
    out.stdout
}

#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn cancel_during_encode_is_reported_as_cancelled() {
    let (w, h, gop, n) = (256u32, 144u32, 4u32, 8u32);
    let yuv = corpus_yuv_concat(w, h, n);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };
    let p = payload::encode_payload("cancelled before the embed", &[]).unwrap();
    let (ct, nonce, salt) = crypto::encrypt(&p, "cancel-pass").unwrap();
    let framed = frame::build_frame(p.len(), &salt, &nonce, &ct);

    progress::cancel();
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let mut progress_calls = 0usize;
    let mut on_progress =
        |_e: phasm_core::codec::av1::stego::encode_progress::Av1EncodeProgress| progress_calls += 1;
    let res = av1_encode_streaming(
        &mut src,
        n,
        params,
        &framed,
        "cancel-pass",
        Pass1CacheBudget::from_env_or_default(),
        Some(&mut on_progress),
    );
    progress::init(0); // clear the flag
    // Observed before the first GOP's rav1e encode, not after Pass 1.
    assert_eq!(progress_calls, 0, "Pass 1 kept encoding after the cancel");

    assert!(
        matches!(res, Err(Av1StegoError::Stego(StegoError::Cancelled))),
        "expected Cancelled, got {:?}",
        res.map(|v| v.len())
    );

    // Same for the shadow path (sequentially — the flag is global).
    let shadow = payload::encode_payload("shadow", &[]).unwrap();
    let shadows: &[(&str, &[u8])] = &[("shadow-pass", &shadow)];
    progress::cancel();
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let res = av1_encode_with_shadows_streaming(
        &mut src,
        n,
        params,
        &framed,
        "cancel-pass",
        shadows,
        16,
        Pass1CacheBudget::from_env_or_default(),
    );
    progress::init(0);
    assert!(
        matches!(res, Err(Av1StegoError::Stego(StegoError::Cancelled))),
        "shadow path: expected Cancelled, got {:?}",
        res.map(|v| v.len())
    );
}

/// A cancel between the passes stops before Pass 2 does any GOP's work: no
/// rebuild, no re-encode, no `Embed` progress event. Run uncached (budget
/// 0 / 0) so Pass 2's first GOP would be a full re-encode from the source —
/// the costly case — and the source log shows whether Pass 2 started one.
#[test]
#[ignore = "AV1 streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn cancel_after_pass1_stops_pass2() {
    use phasm_core::codec::av1::stego::encode_progress::{Av1EncodePhase, Av1EncodeProgress};

    let (w, h, gop, n) = (256u32, 144u32, 4u32, 8u32);
    let yuv = corpus_yuv_concat(w, h, n);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };
    let p = payload::encode_payload("cancelled between the passes", &[]).unwrap();
    let (ct, nonce, salt) = crypto::encrypt(&p, "cancel-pass").unwrap();
    let framed = frame::build_frame(p.len(), &salt, &nonce, &ct);

    let mut src = LoggingSource { inner: Av1SliceYuvSource::new(&yuv, w, h, n, gop), log: Vec::new() };
    let mut after_cancel: Vec<Av1EncodeProgress> = Vec::new();
    let mut cancelled = false;
    let mut on_progress = |e: Av1EncodeProgress| {
        if cancelled {
            after_cancel.push(e);
        } else if e.phase == Av1EncodePhase::Encode && e.gop == e.n_gops {
            progress::cancel();
            cancelled = true;
        }
    };
    let res = av1_encode_streaming(
        &mut src,
        n,
        params,
        &framed,
        "cancel-pass",
        Pass1CacheBudget::new(0, 0, common::scratch_dir().into()),
        Some(&mut on_progress),
    );
    progress::init(0);
    assert!(cancelled, "Pass 1 never reported its last GOP");
    // Pass 1 only: one rewind and both GOPs. A Pass-2 pull would mean a
    // GOP re-encode started after the cancel.
    assert_eq!(src.log, vec![None, Some(0), Some(1)], "Pass 2 read the source after the cancel");
    assert!(
        matches!(res, Err(Av1StegoError::Stego(StegoError::Cancelled))),
        "expected Cancelled, got {:?}",
        res.map(|v| v.len())
    );
    assert!(after_cancel.is_empty(), "Pass 2 kept going after the cancel: {after_cancel:?}");
}

/// A source that logs every call: `None` for a rewind, `Some(g)` for a GOP.
struct LoggingSource<'a> {
    inner: Av1SliceYuvSource<'a>,
    log: Vec<Option<u32>>,
}

impl Av1GopYuvSource for LoggingSource<'_> {
    fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, Av1StegoError> {
        self.log.push(Some(gop_index));
        self.inner.gop_yuv(gop_index)
    }
    fn rewind(&mut self) -> Result<(), Av1StegoError> {
        self.log.push(None);
        self.inner.rewind()
    }
}
