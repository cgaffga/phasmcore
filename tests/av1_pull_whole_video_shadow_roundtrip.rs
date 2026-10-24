// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Whole-video shadow round-trip via the AV1 streaming pull API
//! (`av1_encode_with_shadows_streaming` + `Av1StreamingDecodeSession`).
//!
//! Production gate. Pull-API correctness coverage that replaced the
//! deleted `av1_session_whole_video_shadow` push-API test when the
//! push API was removed (#225/#187, 2026-06-29).
//!
//! Encodes a real-content YUV with a primary + 1 shadow message
//! through the production streaming-pull whole-video shadow entry,
//! then decodes:
//!   - primary via `Av1StreamingDecodeSession::finish_primary_payload`
//!   - shadow  via `finish_shadow_whole_video(shadow_pass)`
//!
//! Both must round-trip. Covers:
//!   - `Av1SliceYuvSource` pull + rewind across the 2-sweep cascade
//!   - `av1_encode_with_shadows_streaming` (Sweep 1 harvest +
//!     Sweep 2 cascade with per-GOP RAM bound)
//!   - Cost-pool position selection for whole-video shadows
//!   - Per-GOP RS parity tier escalation if needed
//!   - Headerless brute-force shadow decode (whole-video union cover)

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use phasm_core::codec::av1::stego::primary_plan::gop_primary_caps;
use phasm_core::codec::av1::stego::session::Av1StreamingDecodeSession;
use phasm_core::codec::av1::stego::whole_video::{
    av1_encode_with_shadows_streaming, Av1GopYuvSource, Av1SliceYuvSource, Pass1CacheBudget,
};
use phasm_core::stego::{crypto, frame, payload};
use phasm_core::{av1_probe_gops_at, Av1StreamingEncodeParams};

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

#[test]
fn pull_whole_video_one_shadow_roundtrip() {
    // Measured 1.37 s wall on the small deterministic fixture
    // (256×144 × 8f × gop=4 × 1 shadow). Formerly `#[ignore]`'d
    // under the assumption of a 30-60 s encode; un-ignored
    // 2026-10-18 so CI exercises the production streaming WV-shadow
    // round-trip (`av1_encode_with_shadows_streaming` + both
    // `Av1StreamingDecodeSession::finish_*` sinks) on every build.
    // Mirror of the deleted `wv_round_trip_gop_size_4_one_shadow_8_frames`
    // fixture (256×144 × 8 frames, gop=4, 1 shadow) — known good
    // cover-capacity ratio for the WV-shadow flow.
    let (w, h, gop, n) = (256, 144, 4, 8);
    let quantizer = 30;
    let primary_text = "pull-API WV-shadow primary — substantive content";
    let primary_pass = "wv-primary-pass";
    let shadow_text = "wv-shadow message";
    let shadow_pass = "wv-shadow-pass";

    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer,
        gop_size: gop,
        total_frames_hint: n,
    };
    let yuv = corpus_yuv_concat(w, h, n);

    // Frame primary
    let primary_payload =
        payload::encode_payload(primary_text, &[]).expect("encode primary payload");
    let (cipher, nonce, salt) =
        crypto::encrypt(&primary_payload, primary_pass).expect("encrypt primary");
    let primary_framed =
        frame::build_frame(primary_payload.len(), &salt, &nonce, &cipher);

    // Shadow message is payload-encoded going in (so `decode_payload`
    // can extract it back to text+files at recovery). Matches the
    // pre-deletion test pattern.
    let shadow_payload =
        payload::encode_payload(shadow_text, &[]).expect("encode shadow payload");
    let shadows: &[(&str, &[u8])] = &[(shadow_pass, &shadow_payload)];

    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let stego = av1_encode_with_shadows_streaming(
        &mut src,
        n,
        params,
        &primary_framed,
        primary_pass,
        shadows,
        16, // parity floor — matches the pre-deletion test
        Pass1CacheBudget::from_env_or_default(),
    )
    .expect("streaming WV-shadow encode");
    assert!(!stego.is_empty(), "encoder produced no bytes");

    // Decode primary
    let mut dec_primary = Av1StreamingDecodeSession::create(primary_pass);
    dec_primary.push_bytes(&stego);
    let recovered_primary = dec_primary
        .finish_primary_payload()
        .expect("decode primary");
    assert_eq!(
        recovered_primary.text, primary_text,
        "recovered primary text mismatch"
    );

    // Decode shadow via whole-video union-cover path
    let dec_shadow = Av1StreamingDecodeSession::create(primary_pass);
    let mut dec_shadow = dec_shadow;
    dec_shadow.push_bytes(&stego);
    let recovered_shadow = dec_shadow
        .finish_shadow_whole_video(shadow_pass)
        .expect("decode shadow (whole-video)");
    assert_eq!(
        recovered_shadow.text, shadow_text,
        "recovered shadow text mismatch"
    );
}

/// Deterministic high-entropy text whose framed primary is as close to
/// `target` bytes as possible without exceeding it (sized from
/// `encode_payload` + the frame overhead, so no Argon2 per probe).
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

/// The shadow path's capacity gate: with a 1-frame trailing GOP, a primary
/// over `n_gops × min(GOP capacity)` — which the even split cannot carry —
/// must round-trip alongside a shadow.
#[test]
#[ignore = "AV1 WV-shadow streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn whole_video_shadow_roundtrip_primary_beyond_even_split() {
    let (w, h, gop, n) = (256u32, 144u32, 8u32, 17u32); // GOPs of 8, 8, 1 frames
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };
    let yuv = corpus_yuv_concat(w, h, n);
    let (primary_pass, shadow_pass) = ("wv-beyond-primary", "shadow-pass");
    let shadow_text = "a 32-byte shadow message, here.";

    let mut probe_src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    probe_src.rewind().expect("rewind");
    let cover: Vec<usize> = av1_probe_gops_at(&mut probe_src, n, params, &[0, 1, 2])
        .expect("probe")
        .into_iter()
        .map(|(_, cb)| cb)
        .collect();
    // Zero-wet caps: a 32-byte shadow occupies < 1 k cover bits, far inside
    // the half-gap margin below.
    let caps = gop_primary_caps(&cover, &[0; 3]);
    let even_ceiling = 3 * caps.iter().map(|c| c.payload_bytes).min().unwrap();
    let sigma: usize = caps.iter().map(|c| c.payload_bytes).sum();
    let primary_text = text_for_framed_len((even_ceiling + sigma) / 2);

    let primary_payload = payload::encode_payload(&primary_text, &[]).expect("primary payload");
    let (cipher, nonce, salt) = crypto::encrypt(&primary_payload, primary_pass).expect("encrypt");
    let primary_framed = frame::build_frame(primary_payload.len(), &salt, &nonce, &cipher);
    assert!(
        primary_framed.len() > even_ceiling,
        "precondition: framed {} must exceed the even split's ceiling {even_ceiling} (cover {cover:?})",
        primary_framed.len()
    );
    let shadow_payload = payload::encode_payload(shadow_text, &[]).expect("shadow payload");
    let shadows: &[(&str, &[u8])] = &[(shadow_pass, &shadow_payload)];

    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let stego = av1_encode_with_shadows_streaming(
        &mut src,
        n,
        params,
        &primary_framed,
        primary_pass,
        shadows,
        16,
        Pass1CacheBudget::from_env_or_default(),
    )
    .expect("streaming WV-shadow encode");

    let mut dec = Av1StreamingDecodeSession::create(primary_pass);
    dec.push_bytes(&stego);
    assert_eq!(dec.finish_primary_payload().expect("decode primary").text, primary_text);
    let mut dec = Av1StreamingDecodeSession::create(primary_pass);
    dec.push_bytes(&stego);
    assert_eq!(
        dec.finish_shadow_whole_video(shadow_pass).expect("decode shadow").text,
        shadow_text
    );
}

/// A GOP too poor to carry even a header-only chunk makes a shadow encode
/// impossible (every GOP carries a chunk + its shadow bits; higher parity
/// tiers only add wet positions). It must fail at once with a clear error,
/// not after sweeping every parity tier and ending in `FrameCorrupted`.
#[test]
#[ignore = "AV1 WV-shadow streaming encode + ffmpeg corpus pull; run with --ignored --test-threads=1"]
fn whole_video_shadow_header_incapable_gop_fails_fast() {
    use phasm_core::codec::av1::stego::orchestrator::Av1StegoError;

    let (w, h, gop, n) = (256u32, 144u32, 1u32, 3u32); // one frame per GOP
    let mut yuv = corpus_yuv_concat(w, h, n);
    let frame_size = (w * h * 3 / 2) as usize;
    yuv[frame_size..2 * frame_size].fill(128); // GOP 1: flat grey
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 30,
        gop_size: gop,
        total_frames_hint: n,
    };
    let primary_payload = payload::encode_payload("primary", &[]).unwrap();
    let (cipher, nonce, salt) = crypto::encrypt(&primary_payload, "p").unwrap();
    let primary_framed = frame::build_frame(primary_payload.len(), &salt, &nonce, &cipher);
    let shadow_payload = payload::encode_payload("shadow", &[]).unwrap();
    let shadows: &[(&str, &[u8])] = &[("s", &shadow_payload)];

    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    let res = av1_encode_with_shadows_streaming(
        &mut src,
        n,
        params,
        &primary_framed,
        "p",
        shadows,
        16,
        Pass1CacheBudget::from_env_or_default(),
    );
    assert!(
        matches!(res, Err(Av1StegoError::InvalidPacket(ref m)) if m.contains("too little cover")),
        "expected the header-incapable-GOP error, got {:?}",
        res.map(|v| v.len())
    );
}
